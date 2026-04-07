use once_cell::sync::Lazy;
use regex::Regex;

static H1_FOLLOWING_DIV_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)(<h1\b[^>]*>.*?</h1>)\s*<div\b[^>]*>(.*?)</div>"#).expect("valid regex")
});
static H1_FOLLOWING_P_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)(<h1\b[^>]*>.*?</h1>)\s*<p\b[^>]*>(.*?)</p>"#).expect("valid regex")
});
static H1_FOLLOWING_SECTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)(<h1\b[^>]*>.*?</h1>)\s*<section\b[^>]*>(.*?)</section>"#)
        .expect("valid regex")
});

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)\b(\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},\s+\d{4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b",
    )
    .expect("valid regex")
});

static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<[^>]+>").expect("valid regex"));

pub fn remove_metadata_block(content_html: &str, has_author_or_published: bool) -> String {
    if !has_author_or_published {
        return content_html.to_string();
    }

    let captures = H1_FOLLOWING_DIV_RE
        .captures(content_html)
        .or_else(|| H1_FOLLOWING_P_RE.captures(content_html))
        .or_else(|| H1_FOLLOWING_SECTION_RE.captures(content_html));
    let Some(caps) = captures else {
        return content_html.to_string();
    };
    let Some(full_match) = caps.get(0) else {
        return content_html.to_string();
    };
    let Some(h1_match) = caps.get(1) else {
        return content_html.to_string();
    };
    let Some(following_match) = caps.get(2) else {
        return content_html.to_string();
    };

    let following_text = strip_tags(following_match.as_str()).trim().to_string();
    if following_text.is_empty() || following_text.len() > 300 {
        return content_html.to_string();
    }
    if !DATE_RE.is_match(&following_text) {
        return content_html.to_string();
    }

    let replacement = h1_match.as_str().to_string();
    let mut out = String::with_capacity(content_html.len());
    out.push_str(&content_html[..full_match.start()]);
    out.push_str(&replacement);
    out.push_str(&content_html[full_match.end()..]);
    out
}

fn strip_tags(input: &str) -> String {
    TAG_RE.replace_all(input, " ").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_h1_adjacent_date_block_when_metadata_present() {
        let html = "<h1>Title</h1><div>By Alice • Mar 29, 2026</div><p>Body</p>";
        let cleaned = remove_metadata_block(html, true);
        assert_eq!(cleaned, "<h1>Title</h1><p>Body</p>");
    }

    #[test]
    fn keeps_following_block_when_no_date() {
        let html = "<h1>Title</h1><div>Important intro block</div><p>Body</p>";
        let cleaned = remove_metadata_block(html, true);
        assert_eq!(cleaned, html);
    }
}
