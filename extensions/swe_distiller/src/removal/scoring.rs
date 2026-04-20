use scraper::{Html, Selector};

use crate::dom::count_words;
use crate::scoring::{is_likely_content, score_non_content_block};

pub fn remove_low_scoring_blocks(input: &str) -> String {
    let html = Html::parse_fragment(input);
    let block_sel =
        Selector::parse("div, section, aside, nav, footer, header").expect("valid selector");
    let mut output = input.to_string();

    for el in html.select(&block_sel) {
        if is_likely_content(&el) {
            continue;
        }
        let text = el.text().collect::<String>();
        if count_words(&text) < 8 {
            continue;
        }
        let score = score_non_content_block(&el);
        if score < 0.0 {
            output = output.replacen(&el.html(), "", 1);
        }
    }

    output
}
