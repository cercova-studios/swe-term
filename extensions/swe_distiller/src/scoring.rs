use scraper::{ElementRef, Selector};

use crate::constants::{CONTENT_INDICATORS, NAVIGATION_INDICATORS};
use crate::dom::count_words;

pub fn score_element(element: &ElementRef<'_>) -> f64 {
    let text = element.text().collect::<String>();
    let words = count_words(&text);
    let mut score = words as f64;

    let p_sel = Selector::parse("p").expect("valid selector");
    score += (element.select(&p_sel).count() as f64) * 10.0;
    score += text.matches(',').count() as f64;

    let img_sel = Selector::parse("img").expect("valid selector");
    let images = element.select(&img_sel).count();
    let image_density = images as f64 / words.max(1) as f64;
    score -= image_density * 3.0;

    let attrs = format!(
        "{} {}",
        element.value().attr("class").unwrap_or_default(),
        element.value().attr("id").unwrap_or_default()
    )
    .to_lowercase();
    if CONTENT_INDICATORS.iter().any(|i| attrs.contains(i)) {
        score += 15.0;
    }

    let link_density = link_density(element, &text);
    score * (1.0 - link_density.min(0.5))
}

pub fn score_non_content_block(element: &ElementRef<'_>) -> f64 {
    let text = element.text().collect::<String>();
    let lower = text.to_lowercase();
    let mut score = 0.0;

    for indicator in NAVIGATION_INDICATORS {
        if lower.contains(indicator) {
            score -= 10.0;
        }
    }

    let ld = link_density(element, &text);
    if ld > 0.5 {
        score -= 15.0;
    }

    let commas = text.matches(',').count() as f64;
    score + commas
}

pub fn is_likely_content(element: &ElementRef<'_>) -> bool {
    if matches!(element.value().attr("role"), Some("article" | "main")) {
        return true;
    }

    let text = element.text().collect::<String>();
    let words = count_words(&text);
    if words > 100 {
        return true;
    }

    let p_sel = Selector::parse("p, li").expect("valid selector");
    let paragraph_like = element.select(&p_sel).count();
    if words > 50 && paragraph_like > 1 {
        return true;
    }

    let code_sel = Selector::parse("pre, table").expect("valid selector");
    element.select(&code_sel).next().is_some()
}

fn link_density(element: &ElementRef<'_>, text: &str) -> f64 {
    let a_sel = Selector::parse("a").expect("valid selector");
    let link_text_len: usize = element
        .select(&a_sel)
        .map(|a| a.text().collect::<String>().len())
        .sum();
    let text_len = text.len().max(1);
    link_text_len as f64 / text_len as f64
}
