use once_cell::sync::Lazy;
use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};

#[derive(Debug, Clone, Copy)]
struct RenderCtx {
    list_depth: usize,
}

pub fn html_to_markdown(html: &str, _url: Option<&str>) -> String {
    let parsed = Html::parse_fragment(html);
    let mut out = String::new();
    let root = parsed.tree.root();
    for child in root.children() {
        render_node(
            child.value(),
            child.children(),
            RenderCtx { list_depth: 0 },
            &mut out,
        );
    }

    crate::markdown::postprocess_markdown(&out)
}

fn render_node<'a, I>(node: &Node, children: I, ctx: RenderCtx, out: &mut String)
where
    I: Iterator<Item = ego_tree::NodeRef<'a, Node>>,
{
    match node {
        Node::Text(t) => push_inline(out, t.text.as_ref()),
        Node::Element(_) => {
            // Element rendering happens in render_element where we have ElementRef.
            for child in children {
                if let Some(el) = ElementRef::wrap(child) {
                    render_element(el, ctx, out);
                } else {
                    render_node(child.value(), child.children(), ctx, out);
                }
            }
        }
        _ => {}
    }
}

fn render_element(el: ElementRef<'_>, ctx: RenderCtx, out: &mut String) {
    let tag = el.value().name();
    match tag {
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
            let level = tag[1..].parse::<usize>().unwrap_or(2).clamp(1, 6);
            let heading = render_inline_children(&el);
            if !heading.trim().is_empty() {
                push_block(out, &format!("{} {}", "#".repeat(level), heading.trim()));
            }
        }
        "p" => {
            let text = render_inline_children(&el);
            if !text.trim().is_empty() {
                push_block(out, text.trim());
            }
        }
        "a" => {
            let href = el.value().attr("href").unwrap_or("#");
            let text = render_inline_children(&el);
            if !text.trim().is_empty() {
                push_block(out, &format!("[{}]({href})", text.trim()));
            }
        }
        "pre" => {
            let code = render_preformatted(&el);
            if !code.trim().is_empty() {
                let lang = detect_code_language(&el).unwrap_or_default();
                if lang.is_empty() {
                    push_block(out, &format!("```\n{}\n```", code.trim_end()));
                } else {
                    push_block(out, &format!("```{lang}\n{}\n```", code.trim_end()));
                }
            }
        }
        "blockquote" => {
            let text = render_inline_children(&el);
            if !text.trim().is_empty() {
                let quoted = text
                    .lines()
                    .map(|line| format!("> {}", line.trim()))
                    .collect::<Vec<_>>()
                    .join("\n");
                push_block(out, &quoted);
            }
        }
        "ul" | "ol" => {
            for (idx, li) in el
                .children()
                .filter_map(ElementRef::wrap)
                .filter(|child| child.value().name() == "li")
                .enumerate()
            {
                let prefix = if tag == "ol" {
                    format!("{}.", idx + 1)
                } else {
                    "-".to_string()
                };
                let indent = "  ".repeat(ctx.list_depth);
                let line = render_inline_children(&li);
                if !line.trim().is_empty() {
                    out.push_str(&format!("{indent}{prefix} {}\n", line.trim()));
                }
                for nested in li
                    .children()
                    .filter_map(ElementRef::wrap)
                    .filter(|child| child.value().name() == "ul" || child.value().name() == "ol")
                {
                    render_element(
                        nested,
                        RenderCtx {
                            list_depth: ctx.list_depth + 1,
                        },
                        out,
                    );
                }
            }
            out.push('\n');
        }
        "table" => {
            if let Some(table_md) = table_to_markdown(&el) {
                push_block(out, &table_md);
            }
        }
        "hr" => push_block(out, "---"),
        "br" => out.push('\n'),
        _ => {
            for child in el.children() {
                if let Some(child_el) = ElementRef::wrap(child) {
                    render_element(child_el, ctx, out);
                } else {
                    render_node(child.value(), child.children(), ctx, out);
                }
            }
        }
    }
}

fn render_inline_children(el: &ElementRef<'_>) -> String {
    let mut out = String::new();
    for child in el.children() {
        if let Some(child_el) = ElementRef::wrap(child) {
            let tag = child_el.value().name();
            match tag {
                "a" => {
                    let href = child_el.value().attr("href").unwrap_or("#");
                    let text = render_inline_children(&child_el);
                    out.push_str(&format!("[{}]({href})", text.trim()));
                }
                "strong" | "b" => {
                    let text = render_inline_children(&child_el);
                    if !text.trim().is_empty() {
                        out.push_str(&format!("**{}**", text.trim()));
                    }
                }
                "em" | "i" => {
                    let text = render_inline_children(&child_el);
                    if !text.trim().is_empty() {
                        out.push_str(&format!("*{}*", text.trim()));
                    }
                }
                "code" => {
                    let text = collect_text(&child_el);
                    if !text.trim().is_empty() {
                        out.push_str(&format!("`{}`", text.trim()));
                    }
                }
                "img" => {
                    let src = child_el.value().attr("src").unwrap_or_default();
                    let alt = child_el.value().attr("alt").unwrap_or_default();
                    out.push_str(&format!("![{alt}]({src})"));
                }
                "br" => out.push('\n'),
                _ => out.push_str(&render_inline_children(&child_el)),
            }
        } else if let Node::Text(t) = child.value() {
            push_inline(&mut out, t.text.as_ref());
        }
    }
    collapse_whitespace(&out)
}

fn render_preformatted(el: &ElementRef<'_>) -> String {
    let mut out = String::new();
    for child in el.children() {
        match child.value() {
            Node::Text(t) => out.push_str(t.text.as_ref()),
            Node::Element(_) => {
                if let Some(inner) = ElementRef::wrap(child) {
                    out.push_str(&collect_text(&inner));
                }
            }
            _ => {}
        }
    }
    out
}

fn table_to_markdown(el: &ElementRef<'_>) -> Option<String> {
    let tr_sel = Selector::parse("tr").ok()?;
    let th_sel = Selector::parse("th").ok()?;
    let td_sel = Selector::parse("td").ok()?;

    let rows: Vec<Vec<String>> = el
        .select(&tr_sel)
        .map(|tr| {
            let mut cells: Vec<String> = tr
                .select(&th_sel)
                .map(|c| collapse_whitespace(&collect_text(&c)))
                .filter(|s| !s.is_empty())
                .collect();
            if cells.is_empty() {
                cells = tr
                    .select(&td_sel)
                    .map(|c| collapse_whitespace(&collect_text(&c)))
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            cells
        })
        .filter(|row| !row.is_empty())
        .collect();
    if rows.is_empty() {
        return None;
    }

    let header = &rows[0];
    let mut lines = vec![format!("| {} |", header.join(" | "))];
    lines.push(format!(
        "| {} |",
        header.iter().map(|_| "---").collect::<Vec<_>>().join(" | ")
    ));
    for row in rows.iter().skip(1) {
        lines.push(format!("| {} |", row.join(" | ")));
    }
    Some(lines.join("\n"))
}

fn collect_text(el: &ElementRef<'_>) -> String {
    let mut out = String::new();
    for child in el.children() {
        if let Some(inner) = ElementRef::wrap(child) {
            out.push_str(&collect_text(&inner));
        } else if let Node::Text(t) = child.value() {
            out.push_str(t.text.as_ref());
        }
    }
    out
}

fn push_block(out: &mut String, block: &str) {
    if block.trim().is_empty() {
        return;
    }
    if !out.is_empty() && !out.ends_with("\n\n") {
        if out.ends_with('\n') {
            out.push('\n');
        } else {
            out.push_str("\n\n");
        }
    }
    out.push_str(block.trim());
    out.push_str("\n\n");
}

fn push_inline(out: &mut String, text: &str) {
    let text = collapse_whitespace(text);
    if text.is_empty() {
        return;
    }
    if !out.is_empty() && !out.ends_with([' ', '\n']) {
        out.push(' ');
    }
    out.push_str(&text);
}

fn collapse_whitespace(input: &str) -> String {
    static WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));
    WS_RE.replace_all(input.trim(), " ").to_string()
}

fn detect_code_language(el: &ElementRef<'_>) -> Option<String> {
    static LANG_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?:^|\\s)(?:language|lang|highlight-source)-([a-zA-Z0-9_+-]+)(?:\\s|$)")
            .expect("valid regex")
    });

    for attr in [
        el.value().attr("class").map(str::to_string),
        first_code_class(el),
    ] {
        if let Some(class_list) = attr.as_deref() {
            if let Some(caps) = LANG_RE.captures(class_list) {
                if let Some(lang) = caps.get(1) {
                    return Some(lang.as_str().to_ascii_lowercase());
                }
            }
        }
    }
    None
}

fn first_code_class(el: &ElementRef<'_>) -> Option<String> {
    let code_sel = Selector::parse("code").ok()?;
    el.select(&code_sel)
        .next()?
        .value()
        .attr("class")
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_nested_inline_formatting() {
        let html = "<p>Hello <strong><em>world</em></strong></p>";
        let md = html_to_markdown(html, None);
        assert!(md.contains("***world***"));
    }

    #[test]
    fn renders_basic_table() {
        let html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>";
        let md = html_to_markdown(html, None);
        assert!(md.contains("| A | B |"));
        assert!(md.contains("| --- | --- |"));
        assert!(md.contains("| 1 | 2 |"));
    }
}
