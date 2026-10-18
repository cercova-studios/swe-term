//! HTML reflow with a separately extracted PDF as a bounded completeness witness.
use std::{collections::BTreeSet, fs, path::Path, time::Duration};

use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use scraper::{ElementRef, Html, Node, Selector};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;
use url::Url;

use crate::{
    dom_ops,
    pdf_reconstruct::Layout,
    types::{DistillerOptions, DistillerResponse},
};

fn normalized(text: &str) -> String {
    text.nfkc()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn label(text: &str) -> Option<String> {
    let mut words = text.split_whitespace();
    let kind = words.next()?.to_ascii_lowercase();
    if !matches!(kind.as_str(), "figure" | "table") {
        return None;
    }
    let number = words.next()?.trim_end_matches(':');
    if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("{kind}-{number}"))
}

fn visible_text(element: ElementRef<'_>) -> String {
    element
        .descendants()
        .filter(|node| {
            !node.ancestors().filter_map(ElementRef::wrap).any(|el| {
                matches!(
                    el.value().name(),
                    "annotation" | "annotation-xml" | "mphantom"
                )
            })
        })
        .filter_map(|node| match node.value() {
            Node::Text(t) => Some(t.text.as_ref()),
            _ => None,
        })
        .collect()
}

fn arxiv_id(url: &Url) -> Option<String> {
    if !matches!(url.host_str(), Some("arxiv.org" | "www.arxiv.org")) {
        return None;
    }
    let path = url
        .path()
        .strip_prefix("/html/")
        .or_else(|| url.path().strip_prefix("/abs/"))
        .or_else(|| url.path().strip_prefix("/pdf/"))?
        .trim_end_matches(".pdf");
    let (number, version) = path.rsplit_once('v')?;
    if number.is_empty()
        || version.is_empty()
        || !version.chars().all(|c| c.is_ascii_digit())
        || !number
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '/' | '-'))
    {
        return None;
    }
    Some(path.into())
}

fn pdf_arxiv_id(layout: &Layout) -> Option<String> {
    let stamp: String = layout
        .spans
        .iter()
        .filter(|s| s.page == 1 && s.rotated && s.bounds[2] < layout.pages[0][0] * 0.08)
        .map(|s| s.text.as_str())
        .collect();
    let pattern =
        regex::Regex::new(r"arXiv:\s*([a-zA-Z.-]+/\d{7}v\d+|\d{4}\.\d{4,5}v\d+)").unwrap();
    let reversed = stamp.chars().rev().collect::<String>();
    pattern
        .captures(&stamp)
        .or_else(|| pattern.captures(&reversed))
        .map(|c| c[1].to_string())
}

async fn download_pdf(url: &Url, path: &Path) -> Result<String> {
    let mut response = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?
        .get(url.clone())
        .header("User-Agent", crate::fetch::DEFAULT_UA)
        .send()
        .await?
        .error_for_status()?;
    ensure!(
        arxiv_id(response.url()) == arxiv_id(url),
        "PDF redirect changed the arXiv version"
    );
    let effective = response.url().to_string();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            bytes.len() + chunk.len() <= 200 * 1024 * 1024,
            "PDF exceeds 200 MiB"
        );
        bytes.extend_from_slice(&chunk);
    }
    ensure!(bytes.starts_with(b"%PDF-"), "Paired source is not a PDF");
    fs::write(path, bytes)?;
    Ok(effective)
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn inline_image(work: &Path, file: &str, alt: &str) -> Result<String> {
    let bytes = fs::read(work.join(file))?;
    ensure!(
        bytes.len() <= 4 * 1024 * 1024,
        "Recovered image exceeds 4 MiB"
    );
    Ok(format!(
        "<img src=\"data:image/png;base64,{}\" alt=\"{}\">",
        STANDARD.encode(bytes),
        escape(alt)
    ))
}

fn relocate_notes(doc: &mut Html) -> Result<String> {
    let notes = Selector::parse(".ltx_note").unwrap();
    let bodies = Selector::parse(".ltx_note_content").unwrap();
    let markers = Selector::parse("sup.ltx_note_mark").unwrap();
    let ids: Vec<_> = doc
        .select(&notes)
        .filter(|n| {
            !n.ancestors().filter_map(ElementRef::wrap).any(|a| {
                a.value()
                    .has_class("ltx_note", scraper::CaseSensitivity::CaseSensitive)
            })
        })
        .map(|n| n.id())
        .collect();
    let mut collected = String::new();
    for (index, id) in ids.into_iter().enumerate() {
        let note = ElementRef::wrap(doc.tree.get(id).unwrap()).unwrap();
        let Some(body) = note.select(&bodies).next() else {
            continue;
        };
        let marker = note
            .select(&markers)
            .next()
            .map(|m| m.text().collect::<String>())
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "*".into());
        let marker_only = body
            .select(&Selector::parse(".ltx_note_type").unwrap())
            .any(|t| {
                t.text()
                    .collect::<String>()
                    .trim()
                    .eq_ignore_ascii_case("footnotemark:")
            });
        let mut content = dom_ops::parse_fragment(&body.inner_html());
        let meta: Vec<_> = content
            .select(&Selector::parse(".ltx_note_type, sup.ltx_note_mark, .ltx_tag_note").unwrap())
            .map(|n| n.id())
            .collect();
        dom_ops::detach_nodes(&mut content, meta);
        let inline_id = note
            .attr("id")
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("distiller-ref-{index}"));
        let note_id = format!("distiller-note-{index}");
        ensure!(
            !doc.select(&Selector::parse("[id]").unwrap())
                .any(|n| n.attr("id") == Some(&note_id)),
            "Generated note ID collides with source"
        );
        if note.attr("id").is_none_or(str::is_empty) {
            ensure!(
                !doc.select(&Selector::parse("[id]").unwrap())
                    .any(|n| n.attr("id") == Some(&inline_id)),
                "Generated reference ID collides with source"
            );
        }
        let body_text = normalized(&visible_text(content.root_element()));
        let marker_only = marker_only && (body_text.is_empty() || body_text == normalized(&marker));
        let replacement = if !marker_only {
            collected += &format!("<li id=\"{note_id}\"><p>{} <a href=\"#{}\" class=\"footnote-back\" role=\"doc-backlink\">↩</a></p></li>",dom_ops::serialize_fragment(&content),escape(&inline_id));
            format!("<a id=\"{}\" href=\"#{note_id}\" class=\"footnote-ref\" role=\"doc-noteref\"><sup>{}</sup></a>",escape(&inline_id),escape(&marker))
        } else {
            // LaTeXML footnotemark tooltips repeat a marker, not a second note body.
            format!(
                "<span id=\"{}\"><sup>{}</sup></span>",
                escape(&inline_id),
                escape(&marker)
            )
        };
        let replacement = dom_ops::parse_fragment(&replacement);
        let element = replacement
            .select(&Selector::parse("a, span").unwrap())
            .next()
            .unwrap();
        let children: Vec<_> = doc
            .tree
            .get(id)
            .unwrap()
            .children()
            .map(|n| n.id())
            .collect();
        dom_ops::detach_nodes(doc, children);
        let mut node = doc.tree.get_mut(id).unwrap();
        *node.value() = Node::Element(element.value().clone());
        let sup = element
            .select(&Selector::parse("sup").unwrap())
            .next()
            .unwrap();
        node.append(Node::Element(sup.value().clone()))
            .append(sup.children().next().unwrap().value().clone());
    }
    Ok(if collected.is_empty() {
        collected
    } else {
        format!("<section class=\"footnotes\" role=\"doc-endnotes\"><ol>{collected}</ol></section>")
    })
}

fn reconcile(raw: &str, layout: &Layout, work: &Path) -> Result<(DistillerResponse, Value)> {
    let mut source = Html::parse_document(raw);
    let article_sel = Selector::parse("article").unwrap();
    let articles: Vec<_> = source.select(&article_sel).collect();
    ensure!(
        articles.len() == 1,
        "PDF-paired HTML requires one unambiguous article element"
    );
    // Keep the complete article, rather than applying readability's prose pruning.
    let mut doc = dom_ops::parse_fragment(&crate::sanitize::sanitize_html(&articles[0].html()));
    let unsupported = Selector::parse("object, svg").unwrap();
    for media in articles[0].select(&unsupported) {
        ensure!(
            media
                .ancestors()
                .filter_map(ElementRef::wrap)
                .any(|el| el.value().name() == "figure"),
            "Unsupported HTML artwork has no figure/caption recovery boundary"
        );
    }
    let unsupported_figures: BTreeSet<_> = articles[0]
        .select(&Selector::parse("figure").unwrap())
        .filter(|f| f.select(&unsupported).next().is_some())
        .map(|f| {
            f.select(&Selector::parse("figcaption").unwrap())
                .next()
                .and_then(|c| label(&c.text().collect::<String>()))
                .context("Unsupported artwork needs a numbered caption")
        })
        .collect::<Result<_>>()?;
    let headings = Selector::parse("h1").unwrap();
    let title = doc
        .select(&headings)
        .next()
        .map(|h| h.text().collect::<String>())
        .unwrap_or_default();
    ensure!(
        !title.trim().is_empty(),
        "PDF-paired HTML needs a document heading"
    );
    let first_page = normalized(
        &layout
            .spans
            .iter()
            .filter(|s| s.page == 1 && !s.clipped && !s.rotated)
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );
    ensure!(
        first_page.contains(&normalized(&title)),
        "HTML title does not match the PDF's first page"
    );
    let heading = doc.select(&headings).next().unwrap().id();
    let article = doc.select(&article_sel).next().unwrap().id();
    // A pre-title license paragraph otherwise causes Pandoc to invent a second title chapter.
    doc.tree.get_mut(article).unwrap().prepend_id(heading);
    let abstracts: Vec<_> = doc
        .select(&Selector::parse(".ltx_title_abstract").unwrap())
        .map(|h| h.id())
        .collect();
    for id in abstracts {
        if let Node::Element(el) = doc.tree.get_mut(id).unwrap().value() {
            el.name.local = "h2".into();
        }
    }
    let auxiliary: Vec<_> = doc
        .select(&Selector::parse(".ltx_title_paragraph").unwrap())
        .map(|h| {
            let rank = h
                .ancestors()
                .filter_map(ElementRef::wrap)
                .filter(|a| a.value().name() == "section")
                .filter_map(|a| {
                    a.children()
                        .filter_map(ElementRef::wrap)
                        .find(|c| {
                            c.id() != h.id()
                                && matches!(
                                    c.value().name(),
                                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                                )
                        })
                        .and_then(|c| c.value().name()[1..].parse::<usize>().ok())
                })
                .next()
                .unwrap_or(1);
            (h.id(), (rank + 1).min(6))
        })
        .collect();
    for (id, rank) in auxiliary {
        if let Node::Element(el) = doc.tree.get_mut(id).unwrap().value() {
            el.name.local = format!("h{rank}").into();
        }
    }
    let text = normalized(&visible_text(doc.root_element()));
    let region_members: BTreeSet<_> = layout
        .regions
        .iter()
        .flat_map(|r| r.spans.iter().copied())
        .collect();
    let mut matched = Vec::new();
    let mut missing = Vec::new();
    let mut excluded = Vec::new();
    let mut short = Vec::new();
    let has_stamp = pdf_arxiv_id(layout).is_some();
    for span in &layout.spans {
        if span.clipped {
            excluded.push(json!({"id":span.id,"reason":"native-clip"}));
        } else if !region_members.contains(&span.id)
            && span.rotated
            && span.page == 1
            && span.bounds[2] < layout.pages[0][0] * 0.08
            && has_stamp
        {
            excluded.push(json!({"id":span.id,"reason":"arxiv-margin-stamp"}));
        } else if span.rotated && !region_members.contains(&span.id) {
            anyhow::bail!("Unresolved rotated PDF text at span {}; native artwork or source review is required",span.id);
        } else if normalized(&span.text).len() < 24 {
            short.push(span.id);
        } else if text.contains(&normalized(&span.text)) {
            matched.push(span.id);
        } else {
            missing.push(span.id);
        }
    }
    ensure!(
        matched.len() >= 3,
        "HTML/PDF pairing has insufficient independent prose witnesses"
    );

    let figures = Selector::parse("figure").unwrap();
    let captions = Selector::parse("figcaption").unwrap();
    let images = Selector::parse("img").unwrap();
    let mut recovered_figures = Vec::new();
    let mut represented_figures = Vec::new();
    let mut used_regions = BTreeSet::new();
    let rendered: Value = serde_json::from_slice(&fs::read(work.join("region-receipt.json"))?)?;
    for caption in layout.spans.iter().filter(|s| label(&s.text).is_some()) {
        let key = label(&caption.text).unwrap();
        // A body reference such as "Table 2 summarizes" is not a caption.
        if !caption
            .text
            .split_whitespace()
            .nth(1)
            .is_some_and(|s| s.ends_with(':'))
        {
            continue;
        }
        let candidates: Vec<_> = doc
            .select(&figures)
            .filter(|f| {
                f.select(&captions)
                    .next()
                    .and_then(|c| label(&c.text().collect::<String>()))
                    .as_deref()
                    == Some(&key)
            })
            .map(|f| f.id())
            .collect();
        ensure!(
            candidates.len() == 1,
            "PDF caption {key} has no unique HTML figure/table destination"
        );
        let figure = candidates[0];
        let element = ElementRef::wrap(doc.tree.get(figure).unwrap()).unwrap();
        if key.starts_with("table-")
            || (element.select(&images).next().is_some() && !unsupported_figures.contains(&key))
        {
            if key.starts_with("table-") {
                ensure!(
                    !unsupported_figures.contains(&key),
                    "Unsupported artwork inside {key} needs source review"
                );
                ensure!(
                    element
                        .select(&Selector::parse("table").unwrap())
                        .next()
                        .is_some(),
                    "HTML {key} lost its table"
                );
            }
            represented_figures.push(key);
            continue;
        }
        let regions: Vec<_> = layout
            .regions
            .iter()
            .filter(|r| {
                r.page == caption.page
                    && !r.graphics.is_empty()
                    && r.bounds[1] >= caption.bounds[3]
                    && !layout.spans.iter().any(|s| {
                        s.page == caption.page
                            && s.id != caption.id
                            && s.y > caption.y
                            && s.y < r.bounds[1]
                            && label(&s.text).is_some()
                    })
            })
            .collect();
        ensure!(
            regions.len() == 1,
            "No unique PDF artwork for {key}; inspect source geometry"
        );
        let caption_node = element.select(&captions).next().unwrap().id();
        let caption_text = visible_text(element.select(&captions).next().unwrap());
        let removed: Vec<_> = element
            .select(&Selector::parse("img, object, svg").unwrap())
            .map(|e| e.id())
            .collect();
        dom_ops::detach_nodes(&mut doc, removed);
        for region in regions {
            ensure!(
                used_regions.insert(region.anchor),
                "Ambiguous shared PDF figure region"
            );
            ensure!(
                !region.spans.contains(&caption.id),
                "PDF crop includes caption already retained in HTML"
            );
            ensure!(
                !layout.spans.iter().any(|s| s.page == caption.page
                    && !s.clipped
                    && !s.rotated
                    && s.bounds[1] > caption.bounds[3]
                    && s.bounds[3] < region.bounds[1]
                    && !region.spans.contains(&s.id)
                    && normalized(&s.text).len() >= 24),
                "Intervening PDF prose makes {key} artwork association ambiguous"
            );
            let image = inline_image(work, &region.file, &caption_text)?;
            let fragment = dom_ops::parse_fragment(&image);
            let node = fragment.select(&images).next().unwrap();
            doc.tree
                .get_mut(caption_node)
                .unwrap()
                .insert_before(Node::Element(node.value().clone()));
            let mut record = rendered["regions"]
                .as_array()
                .context("Missing native crop receipts")?
                .iter()
                .find(|r| r["file"].as_str() == Some(&region.file))
                .context("Missing recovered crop receipt")?
                .clone();
            record["label"] = json!(key);
            record["anchor"] = json!(region.anchor);
            recovered_figures.push(record);
        }
    }

    // Recover only isolated missing source lines with a unique visible DOM neighbour.
    // Larger gaps need semantic reconstruction; do not silently publish an abridgement.
    let paragraphs = Selector::parse("p, li, h1, h2, h3, h4, h5, h6").unwrap();
    let mut recovered_text = Vec::new();
    let mut recovered_notes = String::new();
    let mut insertions = Vec::new();
    for &id in &missing {
        let span = &layout.spans[id];
        if region_members.contains(&id)
            && layout
                .regions
                .iter()
                .any(|r| used_regions.contains(&r.anchor) && r.spans.contains(&id))
        {
            recovered_text.push(id);
            continue;
        }
        ensure!(
            !region_members.contains(&id),
            "Missing native PDF text at span {id}; semantic recovery is unresolved"
        );
        ensure!(
            !missing.iter().any(|&other| other != id
                && layout.spans[other].page == span.page
                && (layout.spans[other].y - span.y).abs() < span.size * 3.),
            "Substantial missing PDF prose near span {id}; HTML completeness is unresolved"
        );
        let mut sizes: Vec<_> = layout
            .spans
            .iter()
            .filter(|s| s.page == span.page && !s.clipped && !s.rotated)
            .map(|s| s.size)
            .collect();
        sizes.sort_by(f32::total_cmp);
        if span.bounds[3] < layout.pages[span.page - 1][1] * 0.12
            && sizes
                .get(sizes.len() / 2)
                .is_some_and(|size| span.size < size * 0.95)
        {
            recovered_notes +=
                &format!("<aside role=\"note\"><p>{}</p></aside>", escape(&span.text));
            recovered_text.push(id);
            continue;
        }
        let (predecessor, targets) = layout.spans[..id]
            .iter()
            .rev()
            .filter(|s| {
                s.page == span.page
                    && matched.contains(&s.id)
                    && (s.x - span.x).abs() <= span.size * 0.75
                    && s.y > span.y
                    && s.y - span.y <= span.size * 6.
            })
            .map(|s| {
                (
                    normalized(&s.text),
                    doc.select(&paragraphs)
                        .filter(|p| normalized(&visible_text(*p)).contains(&normalized(&s.text)))
                        .map(|p| p.id())
                        .collect::<Vec<_>>(),
                )
            })
            .find(|(_, targets)| !targets.is_empty())
            .unwrap_or_default();
        ensure!(
            targets.len() == 1,
            "Missing PDF span {id} has no unique HTML insertion boundary"
        );
        let target = ElementRef::wrap(doc.tree.get(targets[0]).unwrap()).unwrap();
        ensure!(
            normalized(&visible_text(target)).ends_with(&predecessor),
            "Missing PDF span {id} occurs inside an HTML paragraph; recovery order is unresolved"
        );
        ensure!(
            !insertions.contains(&targets[0]),
            "Missing PDF spans share an insertion boundary; recovery order is unresolved"
        );
        insertions.push(targets[0]);
        let fragment = dom_ops::parse_fragment(&format!("<p>{}</p>", escape(&span.text)));
        let p = fragment
            .select(&Selector::parse("p").unwrap())
            .next()
            .unwrap();
        let mut insertion = doc.tree.get_mut(targets[0]).unwrap();
        let mut inserted = insertion.insert_after(Node::Element(p.value().clone()));
        inserted.append(p.children().next().unwrap().value().clone());
        recovered_text.push(id);
    }
    let math = doc.select(&Selector::parse("math").unwrap()).count();
    ensure!(
        math > 0 || !layout.spans.iter().any(|s| s.math && !s.clipped),
        "HTML lost mathematical markup found in the PDF"
    );
    let linked_notes = relocate_notes(&mut doc)?;
    let content = format!(
        "{}{recovered_notes}{linked_notes}",
        dom_ops::serialize_fragment(&doc)
    );
    let author_notes: Vec<_> = source
        .select(&Selector::parse(".ltx_personname .ltx_note").unwrap())
        .map(|n| n.id())
        .collect();
    dom_ops::detach_nodes(&mut source, author_notes);
    let authors = source
        .select(&Selector::parse(".ltx_personname").unwrap())
        .map(|a| a.text().collect::<String>().trim().to_string())
        .collect::<Vec<_>>()
        .join("; ");
    let receipt = json!({"schema":"html-pdf-completeness/v1","pdf_sha256":layout.source_sha256,
        "html_sha256":format!("{:x}",Sha256::digest(raw)), "matched_prose_spans":matched,
        "recovered_text_spans":recovered_text,"recovered_figures":recovered_figures,
        "represented_figures":represented_figures,"excluded_spans":excluded,
        "short_spans_unverified":short,"native_regions":layout.regions.len(),"mathml_expressions":math,
        "native_regions_unverified":layout.regions.iter().filter(|r|!used_regions.contains(&r.anchor)).map(|r|r.anchor).collect::<Vec<_>>(),
        "limits":"Prose witnesses and numbered visual presence are checked. Short spans, equation equivalence, image equivalence and reading order require source inspection."});
    Ok((
        DistillerResponse {
            content,
            title,
            author: authors,
            language: "en".into(),
            ..Default::default()
        },
        receipt,
    ))
}

/// Return false for inputs with no declared PDF/HTML pairing; their existing path stays available.
pub async fn try_write_epub(
    input: &str,
    paired_pdf: Option<&Path>,
    base: Option<&str>,
    output: &Path,
    cover: Option<&Path>,
    opts: &DistillerOptions,
) -> Result<bool> {
    let url = Url::parse(input)
        .ok()
        .filter(|u| matches!(u.scheme(), "http" | "https"));
    let mut id = url.as_ref().and_then(arxiv_id);
    let local_pdf = url.is_none()
        && Path::new(input)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
    if paired_pdf.is_none() && id.is_none() && !local_pdf {
        return Ok(false);
    }
    ensure!(
        !local_pdf
            || (paired_pdf.is_none()
                && base.is_none()
                && opts.proxy_url.is_none()
                && opts.language.is_none()),
        "Local PDF input cannot use --pdf-source, --base-url, --proxy or --lang"
    );
    ensure!(
        url.is_none() || base.is_none(),
        "--base-url is only for HTML files or stdin"
    );
    ensure!(
        !output.exists(),
        "Output already exists: {}",
        output.display()
    );
    let receipt_path = output.with_extension("completeness.json");
    ensure!(
        !receipt_path.exists(),
        "Completeness receipt already exists: {}",
        receipt_path.display()
    );
    let work = tempfile::tempdir()?;
    let prepared = if local_pdf {
        let layout = crate::pdf::extract_layout(Path::new(input), work.path()).await?;
        id = pdf_arxiv_id(&layout);
        if id.is_none() {
            return Ok(false);
        }
        Some(layout)
    } else {
        None
    };
    let paired_pdf = paired_pdf.or_else(|| local_pdf.then_some(Path::new(input)));
    let html_url = id
        .as_ref()
        .map(|id| format!("https://arxiv.org/html/{id}"))
        .or_else(|| url.as_ref().map(ToString::to_string));
    let raw = if let Some(url) = &html_url {
        crate::fetch::fetch_page(
            url,
            crate::fetch::DEFAULT_UA,
            opts.language.as_deref(),
            opts.proxy_url.as_deref(),
            opts.debug,
        )
        .await?
    } else {
        use std::io::Read;
        let mut bytes = Vec::new();
        fs::File::open(input)?
            .take(5 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 5 * 1024 * 1024, "HTML exceeds 5 MiB");
        String::from_utf8(bytes).context("HTML must be valid UTF-8")?
    };
    let downloaded = work.path().join("paired.pdf");
    let pdf_url = if paired_pdf.is_none() {
        Some(
            download_pdf(
                &Url::parse(&format!("https://arxiv.org/pdf/{}", id.as_ref().unwrap()))?,
                &downloaded,
            )
            .await?,
        )
    } else {
        None
    };
    let pdf = paired_pdf.unwrap_or(&downloaded);
    let layout = if let Some(layout) = prepared {
        layout
    } else {
        crate::pdf::extract_layout(pdf, work.path()).await?
    };
    if let Some(id) = &id {
        ensure!(
            pdf_arxiv_id(&layout).as_ref() == Some(id),
            "PDF does not carry the exact requested arXiv version"
        );
        ensure!(
            raw.contains(&format!("/pdf/{id}")),
            "HTML does not declare the requested arXiv PDF version"
        );
    }
    crate::pdf::render_native(pdf, work.path()).await?;
    let (article, mut receipt) = reconcile(&raw, &layout, work.path())?;
    receipt["html_url"] = json!(html_url.as_deref().or(base));
    receipt["pdf_url"] = json!(pdf_url);
    receipt["identity"] = json!(if id.is_some() {
        "exact-arxiv-version-with-prose-witnesses"
    } else {
        "caller-paired-with-title-and-prose-witnesses"
    });
    let local = if url.is_none() && !local_pdf {
        Some(fs::canonicalize(input)?)
    } else {
        None
    };
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let staged = tempfile::tempdir_in(parent)?;
    let staged_book = staged.path().join("article.epub");
    let mut saved = tempfile::NamedTempFile::new_in(parent)?;
    crate::epub::write_epub(
        &article,
        html_url.as_deref().or(base),
        local.as_ref().and_then(|p| p.parent()),
        true,
        &staged_book,
        cover,
    )
    .await?;
    receipt["epub_sha256"] = json!(format!("{:x}", Sha256::digest(fs::read(&staged_book)?)));
    use std::io::Write;
    saved.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    saved.as_file().sync_all()?;
    fs::hard_link(&staged_book, output).context("Publish HTML-first EPUB without overwriting")?;
    saved
        .persist_noclobber(&receipt_path)
        .with_context(||format!("EPUB retained at {}, but its completeness receipt could not be published; do not treat it as verified",output.display()))?;
    eprintln!(
        "HTML-first EPUB with PDF completeness receipt: {}",
        receipt_path.display()
    );
    Ok(true)
}
