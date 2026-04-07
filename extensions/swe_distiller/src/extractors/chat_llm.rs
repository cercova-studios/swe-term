use scraper::{Html, Selector};
use url::Url;

use super::registry::ExtractorResult;

pub fn extract(html: &Html, url: &str) -> Option<ExtractorResult> {
    if !is_chat_share_url(url) {
        return None;
    }

    let selectors = [
        "[data-message-author-role='assistant']",
        "[data-testid='conversation-turn']",
        "article.prose",
        "main",
    ];

    let mut messages = Vec::new();
    for selector in selectors {
        let sel = match Selector::parse(selector) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for node in html.select(&sel) {
            let text = node.text().collect::<String>();
            if text.split_whitespace().count() < 12 {
                continue;
            }
            messages.push(node.html());
        }
        if !messages.is_empty() {
            break;
        }
    }
    if messages.is_empty() {
        return None;
    }

    Some(ExtractorResult {
        content_html: messages.join("\n"),
        site: Some("LLM Chat".to_string()),
        ..ExtractorResult::default()
    })
}

fn is_chat_share_url(url: &str) -> bool {
    Url::parse(url)
        .ok()
        .and_then(|u| {
            u.host_str().map(|h| {
                h.contains("chatgpt.com")
                    || h.ends_with(".openai.com")
                    || h.contains("claude.ai")
                    || h.contains("grok.com")
                    || h.contains("gemini.google.com")
            })
        })
        .unwrap_or(false)
}
