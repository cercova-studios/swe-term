use once_cell::sync::Lazy;
use regex::Regex;
use url::Url;

static DANGEROUS_ELEMENTS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<(?:script|style|noscript|frame|frameset|object|embed|applet|base)\b[^>]*>.*?</(?:script|style|noscript|frame|frameset|object|embed|applet|base)>"#,
    )
    .expect("valid regex")
});
static EVENT_ATTR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)\s+on[a-z]+\s*=\s*(".*?"|'.*?'|[^\s>]+)"#).expect("valid regex")
});
static SRCDOC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\s+srcdoc\s*=\s*(".*?"|'.*?'|[^\s>]+)"#).expect("valid regex"));
static URL_ATTR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)\b(href|src|action|formaction)\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))"#)
        .expect("valid regex")
});

pub fn sanitize_html(input: &str) -> String {
    let mut out = DANGEROUS_ELEMENTS_RE.replace_all(input, "").to_string();
    out = EVENT_ATTR_RE.replace_all(&out, "").to_string();
    out = SRCDOC_RE.replace_all(&out, "").to_string();

    URL_ATTR_RE
        .replace_all(&out, |caps: &regex::Captures<'_>| {
            let attr = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let raw = caps
                .get(3)
                .or_else(|| caps.get(4))
                .or_else(|| caps.get(5))
                .map(|m| m.as_str())
                .unwrap_or_default();
            if is_dangerous_url(raw) {
                String::new()
            } else if caps
                .get(2)
                .map(|m| m.as_str())
                .unwrap_or("")
                .starts_with('"')
            {
                format!(r#"{attr}="{raw}""#)
            } else if caps
                .get(2)
                .map(|m| m.as_str())
                .unwrap_or("")
                .starts_with('\'')
            {
                format!(r#"{attr}='{raw}'"#)
            } else {
                format!(r#"{attr}={raw}"#)
            }
        })
        .to_string()
}

pub fn resolve_relative_urls(input: &str, base_url: &str) -> String {
    let Ok(base) = Url::parse(base_url) else {
        return input.to_string();
    };

    URL_ATTR_RE
        .replace_all(input, |caps: &regex::Captures<'_>| {
            let attr = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            let raw = caps
                .get(3)
                .or_else(|| caps.get(4))
                .or_else(|| caps.get(5))
                .map(|m| m.as_str())
                .unwrap_or_default();

            let resolved = resolve_url(&base, raw).unwrap_or_else(|| raw.to_string());

            if caps
                .get(2)
                .map(|m| m.as_str())
                .unwrap_or("")
                .starts_with('"')
            {
                format!(r#"{attr}="{resolved}""#)
            } else if caps
                .get(2)
                .map(|m| m.as_str())
                .unwrap_or("")
                .starts_with('\'')
            {
                format!(r#"{attr}='{resolved}'"#)
            } else {
                format!(r#"{attr}={resolved}"#)
            }
        })
        .to_string()
}

fn resolve_url(base: &Url, relative: &str) -> Option<String> {
    let trimmed = relative.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("data:") {
        return Some(trimmed.to_string());
    }
    base.join(trimmed).ok().map(|u| u.to_string())
}

pub fn is_dangerous_url(url: &str) -> bool {
    let normalized: String = url
        .chars()
        .filter(|c| !c.is_whitespace() && !c.is_control())
        .collect();
    let lower = normalized.to_lowercase();
    lower.starts_with("javascript:") || lower.starts_with("data:text/html")
}
