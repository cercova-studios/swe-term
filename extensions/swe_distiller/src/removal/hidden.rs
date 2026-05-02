use once_cell::sync::Lazy;
use regex::Regex;

static HIDDEN_STYLE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<(?:div|section|aside|nav|footer|header|span)[^>]*style=["'][^"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0)[^"']*["'][^>]*>.*?</(?:div|section|aside|nav|footer|header|span)>"#,
    )
    .expect("valid regex")
});

static HIDDEN_CLASS_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<(?:div|section|aside|nav|footer|header|span)[^>]*class=["'][^"']*(?:\bhidden\b|\binvisible\b)[^"']*["'][^>]*>.*?</(?:div|section|aside|nav|footer|header|span)>"#,
    )
    .expect("valid regex")
});

pub fn remove_hidden_elements(input: &str) -> String {
    let out = HIDDEN_STYLE_BLOCK_RE.replace_all(input, "").to_string();
    HIDDEN_CLASS_BLOCK_RE.replace_all(&out, "").to_string()
}
