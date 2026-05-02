use once_cell::sync::Lazy;
use regex::Regex;

const MIN_DIMENSION: u32 = 33;

static IMG_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)<(img|svg)\b[^>]*>"#).expect("valid regex"));
static WIDTH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\bwidth=["']?(\d+)"#).expect("valid regex"));
static HEIGHT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\bheight=["']?(\d+)"#).expect("valid regex"));

pub fn remove_small_images(input: &str) -> String {
    IMG_TAG_RE
        .replace_all(input, |caps: &regex::Captures<'_>| {
            let tag = caps.get(0).map(|m| m.as_str()).unwrap_or_default();
            let width = WIDTH_RE
                .captures(tag)
                .and_then(|c| c.get(1))
                .and_then(|m| m.as_str().parse::<u32>().ok());
            let height = HEIGHT_RE
                .captures(tag)
                .and_then(|c| c.get(1))
                .and_then(|m| m.as_str().parse::<u32>().ok());

            if let (Some(w), Some(h)) = (width, height) {
                if w < MIN_DIMENSION || h < MIN_DIMENSION {
                    return String::new();
                }
            }
            tag.to_string()
        })
        .to_string()
}
