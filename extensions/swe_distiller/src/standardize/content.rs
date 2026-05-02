use once_cell::sync::Lazy;
use regex::Regex;

static NBSP_RE: Lazy<Regex> = Lazy::new(|| Regex::new("\u{00A0}").expect("valid regex"));
static COMMENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<!--.*?-->").expect("valid regex"));
static H1_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h1([^>]*)>").expect("valid regex"));
static EMPTY_P_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<p\b[^>]*>\s*</p>").expect("valid regex"));
static EMPTY_DIV_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<div\b[^>]*>\s*</div>").expect("valid regex"));
static EMPTY_SECTION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<section\b[^>]*>\s*</section>").expect("valid regex"));
static EMPTY_ARTICLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<article\b[^>]*>\s*</article>").expect("valid regex"));
static EMPTY_SPAN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<span\b[^>]*>\s*</span>").expect("valid regex"));
static EXCESS_BR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)(<br\s*/?>\s*){3,}").expect("valid regex"));
static WS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]{2,}").expect("valid regex"));

pub fn standardize_content(input: &str, _page_title: &str) -> String {
    let mut out = input.to_string();
    out = NBSP_RE.replace_all(&out, " ").to_string();
    out = COMMENT_RE.replace_all(&out, "").to_string();
    out = H1_RE.replace_all(&out, "<h2$1>").to_string();
    out = out.replace("</h1>", "</h2>");
    out = EMPTY_P_RE.replace_all(&out, "").to_string();
    out = EMPTY_DIV_RE.replace_all(&out, "").to_string();
    out = EMPTY_SECTION_RE.replace_all(&out, "").to_string();
    out = EMPTY_ARTICLE_RE.replace_all(&out, "").to_string();
    out = EMPTY_SPAN_RE.replace_all(&out, "").to_string();
    out = EXCESS_BR_RE.replace_all(&out, "<br><br>").to_string();
    out = WS_RE.replace_all(&out, " ").to_string();
    out.trim().to_string()
}
