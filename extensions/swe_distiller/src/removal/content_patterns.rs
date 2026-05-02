use once_cell::sync::Lazy;
use regex::Regex;

static READ_TIME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b\d+\s*min(?:ute)?s?\s+read\b").expect("valid regex"));
static NEWSLETTER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?is)<(?:p|div|section|aside|footer)[^>]*>[^<]*(?:subscribe|sign\s*up)[^<]*(?:newsletter|email)[^<]*</(?:p|div|section|aside|footer)>",
    )
    .expect("valid regex")
});
static RELATED_HEADING_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)<h[1-6][^>]*>\s*(related|you might also like|read next|further reading|more blog posts to read)[^<]*</h[1-6]>")
        .expect("valid regex")
});
static BOILERPLATE_LINE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?im)^\s*(this (article|story)|originally published|copyright|all rights reserved).*$",
    )
    .expect("valid regex")
});
static SHARE_ARTICLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)\bshare this article\b").expect("valid regex"));
static SHORT_CHROME_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<(?:p|div|section|span|li)[^>]*>\s*(?:sign\s*up|sign\s*in|get\s*app|write|search|follow|listen|share|top\s*highlight)\s*</(?:p|div|section|span|li)>"#,
    )
    .expect("valid regex")
});
static IMAGE_HINT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)\bpress enter or click to view image in full size\b").expect("valid regex")
});
static STANDALONE_COUNTER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<(?:p|div|span|li)[^>]*>\s*(?:[0-9]{1,4}|[·!])\s*</(?:p|div|span|li)>"#)
        .expect("valid regex")
});

pub fn remove_content_patterns(input: &str) -> String {
    let mut out = input.to_string();
    out = READ_TIME_RE.replace_all(&out, "").to_string();
    out = NEWSLETTER_RE.replace_all(&out, "").to_string();
    out = BOILERPLATE_LINE_RE.replace_all(&out, "").to_string();
    out = SHARE_ARTICLE_RE.replace_all(&out, "").to_string();
    out = SHORT_CHROME_BLOCK_RE.replace_all(&out, "").to_string();
    out = IMAGE_HINT_RE.replace_all(&out, "").to_string();
    out = STANDALONE_COUNTER_RE.replace_all(&out, "").to_string();

    if let Some(m) = RELATED_HEADING_RE.find(&out) {
        out.truncate(m.start());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::remove_content_patterns;

    #[test]
    fn removes_short_ui_chrome_blocks() {
        let html = "<p>Sign up</p><p>Follow</p><p>Real content paragraph.</p>";
        let cleaned = remove_content_patterns(html);
        assert!(!cleaned.to_lowercase().contains("sign up"));
        assert!(!cleaned.to_lowercase().contains("follow"));
        assert!(cleaned.contains("Real content paragraph."));
    }

    #[test]
    fn removes_image_hint_and_standalone_counters() {
        let html =
            "<p>Press enter or click to view image in full size</p><p>1</p><p>!</p><p>Body text</p>";
        let cleaned = remove_content_patterns(html);
        assert!(!cleaned.to_lowercase().contains("press enter or click"));
        assert!(cleaned.contains("Body text"));
    }
}
