//! EPUB3 packaging of structured article HTML, using sandboxed Pandoc.
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use scraper::{ElementRef, Node, Selector};
use serde_json::json;
use tempfile::{tempdir, NamedTempFile};
use tokio::process::Command;

use crate::dom_ops::{parse_fragment, serialize_fragment};
use crate::types::DistillerResponse;

const MAX_IMAGE: usize = 4 * 1024 * 1024;
const MAX_IMAGES: usize = 64 * 1024 * 1024;

fn set_attribute(el: &mut scraper::node::Element, name: &str, value: &str) {
    el.attrs.retain(|(key, _)| key.local.as_ref() != name);
    let mut key = el.name.clone();
    key.ns = "".into();
    key.prefix = None;
    key.local = name.into();
    el.attrs.push((key, value.into()));
    el.attrs.sort_by(|a, b| a.0.cmp(&b.0));
}

fn prepare_structure(doc: &mut scraper::Html) {
    let selector = |s| Selector::parse(s).unwrap();
    let tables: Vec<_> = doc.select(&selector("table")).map(|e| e.id()).collect();
    for id in tables {
        let table = ElementRef::wrap(doc.tree.get(id).unwrap()).unwrap();
        // Preserve every source anchor when removing an empty layout table.
        if table.text().all(|t| t.trim().is_empty())
            && table
                .select(&selector("img, math, svg, a, input, canvas, audio, video"))
                .next()
                .is_none()
        {
            let anchors: Vec<_> = std::iter::once(table)
                .chain(table.select(&selector("[id]")))
                .filter(|e| e.attr("id").is_some())
                .map(|e| {
                    let mut anchor = e.value().clone();
                    anchor.name.local = "span".into();
                    anchor.attrs.retain(|(name, _)| name.local.as_ref() == "id");
                    anchor
                })
                .collect();
            for anchor in anchors {
                doc.tree
                    .get_mut(id)
                    .unwrap()
                    .insert_before(Node::Element(anchor));
            }
            doc.tree.get_mut(id).unwrap().detach();
            continue;
        }
        if table
            .value()
            .has_class("ltx_eqn_table", scraper::CaseSensitivity::CaseSensitive)
        {
            if table.attr("role").is_none() {
                if let Node::Element(el) = doc.tree.get_mut(id).unwrap().value() {
                    set_attribute(el, "role", "presentation");
                }
            }
            continue;
        }
        if !table
            .value()
            .has_class("ltx_tabular", scraper::CaseSensitivity::CaseSensitive)
            || table.select(&selector("thead, th, table")).next().is_some()
        {
            continue;
        }
        let rows: Vec<_> = table.select(&selector("tr")).collect();
        let ruled = |row: ElementRef<'_>| {
            let cells: Vec<_> = row.select(&selector("td")).collect();
            !cells.is_empty()
                && cells.iter().all(|c| {
                    c.value()
                        .classes()
                        .any(|v| matches!(v, "ltx_border_t" | "ltx_border_tt"))
                })
        };
        let boundary = rows
            .iter()
            .skip(1)
            .take(3)
            .position(|r| ruled(*r))
            .map(|n| n + 1);
        let Some(boundary) = boundary.filter(|n| ruled(rows[0]) && rows.len() >= n + 2) else {
            continue;
        };
        let span = |cell: ElementRef<'_>, name| {
            cell.attr(name)
                .unwrap_or("1")
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=1000).contains(n))
        };
        let width = |row: ElementRef<'_>| {
            row.select(&selector("td"))
                .try_fold(0usize, |n, c| Some(n + span(c, "colspan")?))
        };
        if width(rows[0]).is_none()
            || width(rows[0]) != width(rows[boundary])
            || rows[..boundary].iter().enumerate().any(|(row, r)| {
                r.select(&selector("td"))
                    .any(|c| span(c, "rowspan").is_none_or(|n| row + n > boundary))
            })
        {
            continue;
        }
        let first: Vec<_> = rows[0].select(&selector("td")).collect();
        let nonempty: Vec<_> = first
            .iter()
            .filter(|c| !c.text().collect::<String>().trim().is_empty())
            .collect();
        let labels = !nonempty.is_empty()
            && nonempty.iter().all(|c| {
                c.text()
                    .collect::<String>()
                    .chars()
                    .any(char::is_alphabetic)
            });
        let bold = |row: ElementRef<'_>| {
            row.select(&selector("td"))
                .filter(|c| !c.text().collect::<String>().trim().is_empty())
                .all(|c| {
                    c.select(&selector("b, strong, .ltx_font_bold"))
                        .next()
                        .is_some()
                })
        };
        let spanning = nonempty
            .iter()
            .filter(|c| c.attr("rowspan").and_then(|s| s.parse::<usize>().ok()) == Some(boundary))
            .count();
        // shortcut: infer only styled LaTeXML label bands; ambiguous unmarked tables retain source markup.
        if !labels
            || !((boundary == 1 && bold(rows[0]) && !bold(rows[boundary]))
                || (boundary > 1 && spanning * 2 > first.len()))
        {
            continue;
        }
        let head_rows: Vec<_> = rows[..boundary].iter().map(|r| r.id()).collect();
        let head = crate::dom_ops::parse_fragment("<table><thead></thead></table>");
        let head = head
            .select(&selector("thead"))
            .next()
            .unwrap()
            .value()
            .clone();
        let head_id = doc
            .tree
            .get_mut(id)
            .unwrap()
            .prepend(Node::Element(head))
            .id();
        for row in head_rows {
            doc.tree.get_mut(head_id).unwrap().append_id(row);
        }
    }
    let cells: Vec<_> = doc
        .select(&selector("th, thead td, td.ltx_th"))
        .map(|e| e.id())
        .collect();
    for id in cells {
        let cell = ElementRef::wrap(doc.tree.get(id).unwrap()).unwrap();
        let in_head = cell
            .ancestors()
            .filter_map(ElementRef::wrap)
            .take_while(|a| a.value().name() != "table")
            .any(|a| a.value().name() == "thead");
        let column = in_head
            || cell
                .value()
                .has_class("ltx_th_column", scraper::CaseSensitivity::CaseSensitive);
        let row = cell
            .value()
            .has_class("ltx_th_row", scraper::CaseSensitivity::CaseSensitive);
        if cell.value().name() == "td"
            && !column
            && !row
            && !cell
                .value()
                .has_class("ltx_th", scraper::CaseSensitivity::CaseSensitive)
        {
            continue;
        }
        let empty = cell.text().all(|t| t.trim().is_empty())
            && cell.select(&selector("img, math")).next().is_none();
        let scope = cell.attr("scope").map(str::to_owned).or_else(|| {
            if column {
                Some(
                    if cell.attr("colspan").is_some_and(|v| v != "1") {
                        "colgroup"
                    } else {
                        "col"
                    }
                    .into(),
                )
            } else if row {
                Some(
                    if cell.attr("rowspan").is_some_and(|v| v != "1") {
                        "rowgroup"
                    } else {
                        "row"
                    }
                    .into(),
                )
            } else {
                None
            }
        });
        if let Node::Element(el) = doc.tree.get_mut(id).unwrap().value() {
            el.name.local = "th".into();
            if empty {
                set_attribute(el, "role", "presentation");
            }
            if let Some(scope) = scope {
                if in_head {
                    set_attribute(el, "scope", &scope);
                } else {
                    // Pandoc may demote body th to td; scope is invalid there.
                    if !empty && el.attr("role").is_none() {
                        match scope.as_str() {
                            "row" | "rowgroup" => set_attribute(el, "role", "rowheader"),
                            "col" | "colgroup" => set_attribute(el, "role", "columnheader"),
                            _ => {}
                        }
                    }
                    el.attrs.retain(|(name, _)| name.local.as_ref() != "scope");
                }
            }
        }
    }
    let mut occupied: std::collections::BTreeSet<_> = doc
        .select(&selector("[id]"))
        .filter_map(|e| e.attr("id").map(str::to_owned))
        .collect();
    let figures: Vec<_> = doc.select(&selector("figure")).map(|e| e.id()).collect();
    for (index, id) in figures.into_iter().enumerate() {
        let figure = ElementRef::wrap(doc.tree.get(id).unwrap()).unwrap();
        let owns = |el: ElementRef<'_>| {
            el.ancestors()
                .filter_map(ElementRef::wrap)
                .find(|a| a.value().name() == "figure")
                .is_some_and(|a| a.id() == id)
        };
        let Some(caption) = figure.select(&selector("figcaption")).find(|e| owns(*e)) else {
            continue;
        };
        let text = caption.text().collect::<String>();
        if text.trim().is_empty() {
            continue;
        }
        let caption_node = caption.id();
        let mut key = caption.attr("id").unwrap_or("").to_owned();
        if key.is_empty() {
            key = format!("distiller-caption-{index}");
            while !occupied.insert(key.clone()) {
                key.push('-');
            }
        }
        let children: Vec<_> = caption.children().map(|n| n.id()).collect();
        let mut wrapper = caption.value().clone();
        // Pandoc retains this ID around both inline and block caption content.
        wrapper.name.local = "div".into();
        wrapper.attrs.clear();
        set_attribute(&mut wrapper, "id", &key);
        let images: Vec<_> = figure
            .select(&selector("img, table"))
            .filter(|e| owns(*e))
            .map(|e| e.id())
            .collect();
        if let Node::Element(el) = doc.tree.get_mut(caption_node).unwrap().value() {
            el.attrs.retain(|(n, _)| n.local.as_ref() != "id");
        }
        let wrapper = doc
            .tree
            .get_mut(caption_node)
            .unwrap()
            .append(Node::Element(wrapper))
            .id();
        for child in children {
            doc.tree.get_mut(wrapper).unwrap().append_id(child);
        }
        for (panel, image) in images.into_iter().enumerate() {
            if let Node::Element(el) = doc.tree.get_mut(image).unwrap().value() {
                if el.attr("aria-describedby").is_none()
                    && !matches!(el.attr("role"), Some("presentation" | "none"))
                {
                    set_attribute(el, "aria-describedby", &key);
                }
                if el.name() == "img"
                    && el.attr("alt").is_none_or(|a| {
                        a.trim().is_empty()
                            || matches!(
                                a,
                                "Refer to caption"
                                    | "Figure"
                                    | "Source mathematical or graphical passage"
                            )
                            || a == text.trim()
                    })
                {
                    set_attribute(
                        el,
                        "alt",
                        &format!("Illustration {}, panel {}", index + 1, panel + 1),
                    );
                }
            }
        }
    }
}

pub(crate) fn decode_image(bytes: &[u8]) -> Result<image::DynamicImage> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(12000);
    limits.max_image_height = Some(12000);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .context("Image cannot be decoded within EPUB limits")
}

/// Embed validated local/inline raster images and atomically create a new EPUB.
/// Local resources must stay beneath resource_root, including symlink targets.
pub async fn write_epub(
    article: &DistillerResponse,
    source_url: Option<&str>,
    resource_root: Option<&Path>,
    allow_remote: bool,
    destination: &Path,
    cover: Option<&Path>,
) -> Result<()> {
    ensure!(
        !destination.exists(),
        "Output already exists: {}",
        destination.display()
    );
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent).context("Output directory does not exist")?;
    let output = NamedTempFile::new_in(parent).context("Create temporary EPUB output")?;
    if let Some(source) = source_url {
        let url = url::Url::parse(source).context("Invalid EPUB source URL")?;
        ensure!(
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none(),
            "EPUB source URL must be HTTP(S) without credentials"
        );
    }
    let work = tempdir().context("Create EPUB workspace")?;
    let mut total = 0;
    let mut cover = cover
        .map(|path| -> Result<_> {
            ensure!(
                fs::metadata(path)?.is_file(),
                "Cover must be a regular file"
            );
            let mut bytes = Vec::new();
            File::open(path)?
                .take((MAX_IMAGE + 1) as u64)
                .read_to_end(&mut bytes)?;
            ensure!(bytes.len() <= MAX_IMAGE, "Cover exceeds 4 MiB");
            let extension = match image::guess_format(&bytes)? {
                image::ImageFormat::Png => "png",
                image::ImageFormat::Jpeg => "jpg",
                _ => bail!("Cover must be PNG or JPEG"),
            };
            decode_image(&bytes)?;
            total = bytes.len();
            let saved = work.path().join(format!("cover.{extension}"));
            fs::write(&saved, bytes)?;
            Ok(saved)
        })
        .transpose()?;
    let mut doc = parse_fragment(&crate::sanitize::sanitize_html(&article.content));
    let unsupported = Selector::parse("mglyph, video, audio, canvas, svg").unwrap();
    ensure!(
        doc.select(&unsupported).next().is_none(),
        "EPUB export cannot preserve embedded media, canvas, SVG or MathML image glyphs"
    );
    prepare_structure(&mut doc);
    // Pandoc omits headings inside Div blocks from navigation; flatten their wrappers.
    let headings = Selector::parse("h1, h2, h3, h4, h5, h6").unwrap();
    let wrappers = Selector::parse("div, section, article, main").unwrap();
    let wrappers: Vec<_> = doc
        .select(&wrappers)
        .filter(|el| el.select(&headings).next().is_some())
        .filter(|el| {
            !el.ancestors()
                .filter_map(ElementRef::wrap)
                .any(|ancestor| ancestor.value().name() == "figcaption")
        })
        .map(|el| el.id())
        .collect();
    for id in wrappers {
        let node = doc.tree.get(id).unwrap();
        let children: Vec<_> = node.children().map(|child| child.id()).collect();
        let Node::Element(el) = node.value() else {
            unreachable!()
        };
        ensure!(
            el.attr("lang").is_none() && el.attr("dir").is_none(),
            "EPUB heading wrappers with language/direction attributes cannot yet be flattened safely"
        );
        let mut anchor = el.clone();
        anchor.name.local = "span".into();
        anchor.attrs.retain(|(name, _)| name.local.as_ref() == "id");
        let mut node = doc.tree.get_mut(id).unwrap();
        if !anchor.attrs.is_empty() {
            node.insert_before(Node::Element(anchor));
        }
        for child in children {
            node.insert_id_before(child);
        }
        node.detach();
    }
    // Pandoc drops these container IDs; retain them as inline anchors.
    let anchors = Selector::parse("p[id], li[id], figure[id], table[id]").unwrap();
    let anchor_ids: Vec<_> = doc.select(&anchors).map(|el| el.id()).collect();
    for id in anchor_ids {
        if doc
            .tree
            .get(id)
            .unwrap()
            .ancestors()
            .filter_map(ElementRef::wrap)
            .any(|el| {
                el.attr("role") == Some("doc-endnotes")
                    || el
                        .value()
                        .has_class("footnotes", scraper::CaseSensitivity::CaseSensitive)
            })
        {
            continue;
        }
        let mut node = doc.tree.get_mut(id).unwrap();
        let Node::Element(el) = node.value() else {
            unreachable!()
        };
        let mut anchor = el.clone();
        anchor.name.local = "span".into();
        anchor.attrs.retain(|(name, _)| name.local.as_ref() == "id");
        el.attrs.retain(|(name, _)| name.local.as_ref() != "id");
        if matches!(el.name(), "figure" | "table") {
            node.insert_before(Node::Element(anchor));
        } else {
            node.prepend(Node::Element(anchor));
        }
    }
    let links = Selector::parse("a[href]").unwrap();
    let empty_mail: Vec<_> = doc
        .select(&links)
        .filter(|a| {
            a.attr("href")
                .is_some_and(|h| h.eq_ignore_ascii_case("mailto:"))
        })
        .map(|a| a.id())
        .collect();
    for id in empty_mail {
        if let Node::Element(el) = doc.tree.get_mut(id).unwrap().value() {
            el.attrs.retain(|(name, _)| name.local.as_ref() != "href");
        }
    }
    let images = Selector::parse("img").unwrap();
    let ids: Vec<_> = doc.select(&images).map(|el| el.id()).collect();
    let root = resource_root.map(fs::canonicalize).transpose()?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?;
    for id in ids {
        let node = doc.tree.get(id).unwrap();
        let el = ElementRef::wrap(node).unwrap();
        let source = el
            .value()
            .attr("src")
            .filter(|s| !s.is_empty())
            .or_else(|| el.value().attr("data-src"))
            .or_else(|| el.value().attr("data-original"))
            .context("Image has no src or supported lazy-loading source")?;
        let mut bytes = if source.starts_with("data:") {
            let (header, body) = source.split_once(',').context("Invalid inline image")?;
            ensure!(header.ends_with(";base64"), "Inline images must use base64");
            ensure!(body.len() <= MAX_IMAGE * 4 / 3 + 4, "Image exceeds 4 MiB");
            STANDARD.decode(body).context("Invalid base64 image")?
        } else {
            let relative = source.split(['?', '#']).next().unwrap();
            let local = root
                .as_ref()
                .filter(|_| !source.starts_with('/') && url::Url::parse(source).is_err());
            let path = local
                .map(|root| url::Url::from_directory_path(root).unwrap().join(relative))
                .transpose()?
                .and_then(|u| u.to_file_path().ok());
            if let Some(path) = path.filter(|p| p.exists()) {
                let path = fs::canonicalize(path)?;
                ensure!(
                    path.starts_with(root.as_ref().unwrap()),
                    "Image escapes the HTML resource directory"
                );
                ensure!(path.is_file(), "Image must be a regular file");
                let mut bytes = Vec::new();
                File::open(path)?
                    .take((MAX_IMAGE + 1) as u64)
                    .read_to_end(&mut bytes)?;
                bytes
            } else {
                ensure!(
                    allow_remote,
                    "Stdin EPUB images must be inline data; no fetching is permitted"
                );
                let url = if let Some(base) = source_url {
                    url::Url::parse(base)?.join(source)?
                } else {
                    url::Url::parse(source)
                        .context("Image file is missing and no source URL can resolve it")?
                };
                ensure!(
                    matches!(url.scheme(), "http" | "https")
                        && url.host_str().is_some()
                        && url.username().is_empty()
                        && url.password().is_none(),
                    "Image URL must be HTTP(S) without credentials"
                );
                let mut response = client
                    .get(url)
                    .header("User-Agent", crate::fetch::DEFAULT_UA)
                    .send()
                    .await?
                    .error_for_status()?;
                ensure!(
                    response
                        .content_length()
                        .is_none_or(|n| n <= MAX_IMAGE as u64),
                    "Image exceeds 4 MiB"
                );
                let mut bytes = Vec::new();
                while let Some(chunk) = response.chunk().await? {
                    ensure!(
                        bytes.len() + chunk.len() <= MAX_IMAGE,
                        "Image exceeds 4 MiB"
                    );
                    bytes.extend_from_slice(&chunk);
                }
                bytes
            }
        };
        ensure!(bytes.len() <= MAX_IMAGE, "Image exceeds 4 MiB");
        let decoded = decode_image(&bytes)?;
        if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
            let mut output = Cursor::new(Vec::new());
            decoded.write_to(&mut output, image::ImageFormat::Png)?;
            bytes = output.into_inner();
            ensure!(bytes.len() <= MAX_IMAGE, "Decoded image exceeds 4 MiB");
        }
        total += bytes.len();
        ensure!(total <= MAX_IMAGES, "Embedded images exceed 64 MiB total");
        let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            "image/png"
        } else if bytes.starts_with(b"\xff\xd8\xff") {
            "image/jpeg"
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            "image/gif"
        } else {
            bail!("Unsupported image bytes; EPUB currently supports PNG, JPEG and GIF");
        };
        let mut node = doc.tree.get_mut(id).unwrap();
        let Node::Element(el) = node.value() else {
            unreachable!()
        };
        let mut name = el
            .attrs
            .iter()
            .find(|(name, _)| matches!(name.local.as_ref(), "src" | "data-src" | "data-original"))
            .unwrap()
            .0
            .clone();
        name.local = "src".into();
        el.attrs.retain(|(name, _)| {
            !matches!(
                name.local.as_ref(),
                "src" | "srcset" | "data-src" | "data-original"
            )
        });
        el.attrs.push((
            name,
            format!("data:{mime};base64,{}", STANDARD.encode(bytes)).into(),
        ));
    }
    let mut html = serialize_fragment(&doc);
    if let Some(base) = source_url {
        html = crate::sanitize::resolve_relative_urls(&html, base);
    }
    let title = if !article.title.trim().is_empty() {
        article.title.clone()
    } else {
        doc.select(&Selector::parse("h1").unwrap())
            .next()
            .map(|h| h.text().collect::<String>())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_default()
    };
    ensure!(!title.trim().is_empty(), "EPUB needs an article title");
    if cover.is_none() {
        let mut probe = Command::new("pandoc");
        probe
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let status = tokio::time::timeout(Duration::from_secs(5), probe.status())
            .await
            .context("Pandoc preflight timed out")?
            .context("EPUB export requires Pandoc on PATH")?;
        ensure!(
            status.success(),
            "Pandoc preflight failed; cover generation was not attempted"
        );
        // Reserve the full cover allowance before any paid request.
        ensure!(
            total + MAX_IMAGE <= MAX_IMAGES,
            "No room for a generated cover within the 64 MiB image limit"
        );
        let saved = work.path().join("cover.png");
        let hints = doc
            .select(&headings)
            .take(6)
            .map(|h| {
                h.text()
                    .collect::<String>()
                    .chars()
                    .take(160)
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let bytes = crate::cover::resolve(&title, &article.author, &article.content, hints).await?;
        decode_image(&bytes)?;
        fs::write(&saved, bytes)?;
        cover = Some(saved);
    }
    let input = work.path().join("article.html");
    let header = work.path().join("header.html");
    let metadata = work.path().join("metadata.json");
    fs::write(&input, html)?;
    fs::write(
        &header,
        format!("<style>{}</style>", include_str!("../assets/epub.css")),
    )?;
    let mut meta = json!({"title": title});
    for (key, value) in [
        ("author", article.author.as_str()),
        ("lang", article.language.as_str()),
        ("description", article.description.as_str()),
        ("source", source_url.unwrap_or_default()),
    ] {
        if !value.is_empty() {
            meta[key] = json!(value);
        }
    }
    let published = article.published.trim();
    let date = chrono::DateTime::parse_from_rfc3339(published)
        .ok()
        .map(|d| d.date_naive())
        .or_else(|| {
            ["%Y-%m-%d", "%b %e, %Y", "%B %e, %Y"]
                .iter()
                .find_map(|format| chrono::NaiveDate::parse_from_str(published, format).ok())
        });
    if let Some(date) = date {
        meta["date"] = json!(date.to_string());
    }
    fs::write(&metadata, serde_json::to_vec(&meta)?)?;
    let errors = work.path().join("pandoc.stderr");
    let mut command = Command::new("pandoc");
    if let Some(cover) = cover {
        // shortcut: Pandoc 3.12.1 needs reference-doc to preload this cover; remove when its cover option preloads correctly.
        command.arg("--reference-doc").arg(&cover);
        command.arg("--epub-cover-image").arg(cover);
    }
    command
        .args([
            "--from=html",
            "--to=epub3",
            "--sandbox",
            "--fail-if-warnings",
            "--toc",
            "--toc-depth=6",
            "--split-level=2",
            "--preserve-tabs",
            "--math-method=mathml",
            "--syntax-highlighting=none",
        ])
        .arg("--data-dir")
        .arg(work.path())
        .arg("--include-in-header")
        .arg(&header)
        .arg("--metadata-file")
        .arg(&metadata)
        .arg("--output")
        .arg(output.path())
        .arg(&input)
        .current_dir(work.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(File::create(&errors)?)
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .context("EPUB export requires Pandoc on PATH")?;
    let status = match tokio::time::timeout(Duration::from_secs(30), child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            child.kill().await?;
            bail!("Pandoc exceeded the 30-second EPUB timeout");
        }
    };
    if !status.success() {
        let mut error = String::new();
        File::open(errors)?.take(8192).read_to_string(&mut error)?;
        bail!("Pandoc EPUB export failed: {}", error.trim());
    }
    ensure!(
        output.as_file().metadata()?.len() > 0,
        "Pandoc produced an empty EPUB"
    );
    output
        .persist_noclobber(destination)
        .context("Publish EPUB without overwriting an existing file")?;
    Ok(())
}
