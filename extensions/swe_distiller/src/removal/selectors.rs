use once_cell::sync::Lazy;
use regex::Regex;

use crate::constants::{EXACT_SELECTORS, PARTIAL_PATTERNS};

static BLOCK_TAG_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<(?:nav|aside|footer|header)[^>]*>.*?</(?:nav|aside|footer|header)>")
        .expect("valid regex")
});

pub fn remove_by_selectors(input: &str, remove_exact: bool, remove_partial: bool) -> String {
    let mut out = input.to_string();

    if remove_exact {
        for sel in EXACT_SELECTORS {
            // String-first cleanup keeps implementation simple and safe for CLI mode.
            if *sel == "script:not([type^=\"math/\"])" {
                out = Regex::new(r"(?is)<script[^>]*>.*?</script>")
                    .expect("valid regex")
                    .replace_all(&out, "")
                    .to_string();
            } else if *sel == "style" {
                out = Regex::new(r"(?is)<style[^>]*>.*?</style>")
                    .expect("valid regex")
                    .replace_all(&out, "")
                    .to_string();
            } else if *sel == "noscript" {
                out = Regex::new(r"(?is)<noscript[^>]*>.*?</noscript>")
                    .expect("valid regex")
                    .replace_all(&out, "")
                    .to_string();
            }
        }
        out = BLOCK_TAG_RE.replace_all(&out, "").to_string();
    }

    if remove_partial {
        for pattern in PARTIAL_PATTERNS {
            let re = Regex::new(&format!(
                r#"(?is)<(?:div|section|aside|nav|footer|header)[^>]*(?:class|id|data-[^=]+)=["'][^"']*{pattern}[^"']*["'][^>]*>.*?</(?:div|section|aside|nav|footer|header)>"#
            ))
            .expect("valid regex");
            out = re.replace_all(&out, "").to_string();
        }
    }

    out
}
