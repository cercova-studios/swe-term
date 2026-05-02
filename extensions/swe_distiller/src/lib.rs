pub mod constants;
pub mod dom;
pub mod extraction;
pub mod extractors;
pub mod fetch;
pub mod find_content;
pub mod markdown;
pub mod markdown_ast;
pub mod metadata;
pub mod observability;
pub mod removal;
pub mod sanitize;
pub mod scoring;
pub mod standardize;
pub mod types;

use std::time::Instant;

use crate::extractors::registry::ExtractorResult;
use anyhow::Result;
use scraper::Html;
use types::{DistillerOptions, DistillerResponse};

pub async fn extract_url(url: &str, opts: DistillerOptions) -> Result<DistillerResponse> {
    observability::debug(opts.debug, "pipeline.start", format!("url={url}"));
    let ua = fetch::get_initial_ua(url);
    let html = fetch::fetch_page(
        url,
        ua,
        opts.language.as_deref(),
        opts.proxy_url.as_deref(),
        opts.debug,
    )
    .await?;
    let parsed = Html::parse_document(&html);
    let async_extractor = extractors::registry::run_extractors_async_preferred(&parsed, url).await;
    observability::debug(
        opts.debug,
        "pipeline.extractor",
        if async_extractor.is_some() {
            "async extractor provided content"
        } else {
            "no async extractor match"
        },
    );
    let llm_markdown = if opts.llm {
        extraction::llm::extract_via_llm(&html, url, opts.debug).await
    } else {
        None
    };

    let mut result =
        parse_html_with_extractor(&html, Some(url), opts.clone(), async_extractor.clone());

    if opts.llm {
        if let Some(md_raw) = llm_markdown {
            let title_hint = result.title.clone();
            apply_llm_markdown_override(&mut result, &title_hint, &md_raw);
            observability::debug(
                opts.debug,
                "pipeline.llm",
                format!("outcome={:?}", extraction::llm::outcome_for(true)),
            );
        } else {
            observability::debug(
                opts.debug,
                "pipeline.llm",
                format!("outcome={:?}", extraction::llm::outcome_for(false)),
            );
        }
    }
    observability::debug(
        opts.debug,
        "pipeline.done",
        format!(
            "word_count={} parse_time_ms={}",
            result.word_count, result.parse_time_ms
        ),
    );
    Ok(result)
}

pub fn parse_html(html_str: &str, url: Option<&str>, opts: DistillerOptions) -> DistillerResponse {
    parse_html_with_extractor(html_str, url, opts, None)
}

pub async fn parse_html_async(
    html_str: &str,
    url: Option<&str>,
    opts: DistillerOptions,
) -> DistillerResponse {
    let parsed = Html::parse_document(html_str);
    let extractor_override = match url {
        Some(u) => extractors::registry::run_extractors_async_preferred(&parsed, u).await,
        None => None,
    };
    parse_html_with_extractor(html_str, url, opts, extractor_override)
}

#[derive(Debug, Clone)]
struct RecoveryPlan {
    minimum_word_count: usize,
    options: DistillerOptions,
    require_double_improvement: bool,
}

struct ParsePipeline<'a> {
    html_str: &'a str,
    url: Option<&'a str>,
    opts: DistillerOptions,
    extractor_override: Option<ExtractorResult>,
}

impl<'a> ParsePipeline<'a> {
    fn new(
        html_str: &'a str,
        url: Option<&'a str>,
        opts: DistillerOptions,
        extractor_override: Option<ExtractorResult>,
    ) -> Self {
        Self {
            html_str,
            url,
            opts,
            extractor_override,
        }
    }

    fn run(&self) -> DistillerResponse {
        let start = Instant::now();
        let mut result = self.run_pass(&self.opts);

        for plan in recovery_plans(&self.opts) {
            if result.word_count < plan.minimum_word_count {
                let retry = self.run_pass(&plan.options);
                let improved = if plan.require_double_improvement {
                    retry.word_count > result.word_count.saturating_mul(2)
                } else {
                    retry.word_count > result.word_count
                };
                if improved {
                    result = retry;
                }
            }
        }

        result = self.apply_schema_override(result);
        result.parse_time_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn run_pass(&self, opts: &DistillerOptions) -> DistillerResponse {
        parse_internal(
            self.html_str,
            self.url,
            opts,
            self.extractor_override.as_ref(),
        )
    }

    fn apply_schema_override(&self, result: DistillerResponse) -> DistillerResponse {
        if let Some(schema_text) = metadata::schema_text(result.schema_org_data.as_ref()) {
            let schema_words = dom::count_words(&schema_text);
            if schema_words > (result.word_count * 3) / 2 {
                let mut schema_result = result.clone();
                schema_result.content = schema_text;
                schema_result.word_count = schema_words;
                schema_result.content_markdown =
                    Some(markdown::html_to_markdown(&schema_result.content, self.url));
                return schema_result;
            }
        }

        result
    }
}

fn parse_html_with_extractor(
    html_str: &str,
    url: Option<&str>,
    opts: DistillerOptions,
    extractor_override: Option<ExtractorResult>,
) -> DistillerResponse {
    ParsePipeline::new(html_str, url, opts, extractor_override).run()
}

fn parse_internal(
    html_str: &str,
    url: Option<&str>,
    opts: &DistillerOptions,
    extractor_override: Option<&ExtractorResult>,
) -> DistillerResponse {
    let parsed = Html::parse_document(html_str);

    let meta_tags = metadata::collect_meta_tags(&parsed);
    let schema_org_data = metadata::extract_schema_org(&parsed);
    let extractor_result = extractor_override
        .cloned()
        .or_else(|| url.and_then(|u| extractors::registry::run_extractors(&parsed, u)));

    let mut content_html = extractor_result
        .as_ref()
        .map(|r| r.content_html.clone())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| find_content::find_main_content_html(&parsed))
        .or_else(|| find_content::body_fallback_html(&parsed))
        .unwrap_or_default();

    content_html = standardize::footnotes::standardize_footnotes(&content_html);
    content_html = standardize::callouts::standardize_callouts(&content_html);

    if opts.remove_exact_selectors || opts.remove_partial_selectors {
        content_html = apply_removal_step(
            opts.debug,
            "selectors",
            "exact+partial selector pruning",
            &content_html,
            |input| {
                removal::selectors::remove_by_selectors(
                    input,
                    opts.remove_exact_selectors,
                    opts.remove_partial_selectors,
                )
            },
        );
    }

    if opts.remove_hidden_elements {
        content_html = apply_removal_step(
            opts.debug,
            "hidden",
            "hidden style/attribute pruning",
            &content_html,
            removal::hidden::remove_hidden_elements,
        );
    }

    if opts.remove_small_images {
        content_html = apply_removal_step(
            opts.debug,
            "small_images",
            "small media pruning",
            &content_html,
            removal::small_images::remove_small_images,
        );
    }

    if opts.remove_low_scoring {
        content_html = apply_removal_step(
            opts.debug,
            "scoring",
            "low-scoring block pruning",
            &content_html,
            removal::scoring::remove_low_scoring_blocks,
        );
    }

    if opts.remove_content_patterns {
        content_html = apply_removal_step(
            opts.debug,
            "content_patterns",
            "known clutter-pattern pruning",
            &content_html,
            removal::content_patterns::remove_content_patterns,
        );
    }

    let (author_hint_value, published_hint_value) = author_hint(&parsed, schema_org_data.as_ref());
    let has_author_or_published = !author_hint_value.is_empty() || !published_hint_value.is_empty();
    content_html = apply_removal_step(
        opts.debug,
        "metadata_block",
        "h1-adjacent author/date metadata removal",
        &content_html,
        |input| removal::metadata_block::remove_metadata_block(input, has_author_or_published),
    );

    content_html = standardize::content::standardize_content(&content_html, "");

    if let Some(base_url) = url {
        content_html = sanitize::resolve_relative_urls(&content_html, base_url);
    }

    content_html = sanitize::sanitize_html(&content_html);
    let word_count = dom::count_words(&dom::strip_tags(&content_html));

    let mut md = Some(markdown_ast::html_to_markdown(&content_html, url));

    let md_title_hint = md.as_ref().and_then(|m| markdown::guess_title(m));

    let extracted_title =
        metadata::extract_title(&parsed, schema_org_data.as_ref(), md_title_hint.as_deref());
    let title = choose_override(
        extractor_result.as_ref().and_then(|r| r.title.as_deref()),
        extracted_title,
    );
    if let Some(md_text) = md.as_mut() {
        *md_text = markdown::ensure_title_heading(md_text, &title);
    }

    let md_ref = md.as_deref();
    let description = choose_override(
        extractor_result
            .as_ref()
            .and_then(|r| r.description.as_deref()),
        metadata::extract_description(&parsed, schema_org_data.as_ref(), md_ref),
    );
    let author = choose_override(
        extractor_result.as_ref().and_then(|r| r.author.as_deref()),
        metadata::extract_author(&parsed, schema_org_data.as_ref(), md_ref),
    );
    let published = choose_override(
        extractor_result
            .as_ref()
            .and_then(|r| r.published.as_deref()),
        metadata::extract_published(&parsed, schema_org_data.as_ref(), md_ref),
    );
    let site = choose_override(
        extractor_result.as_ref().and_then(|r| r.site.as_deref()),
        metadata::extract_site(&parsed, schema_org_data.as_ref(), url),
    );
    let image = metadata::extract_image(&parsed, schema_org_data.as_ref());
    let favicon = metadata::extract_favicon(&parsed, url);
    let language = metadata::extract_language(&parsed);
    let domain = metadata::extract_domain(url);

    DistillerResponse {
        content: content_html,
        title,
        description,
        domain,
        favicon,
        image,
        language,
        parse_time_ms: 0,
        published,
        author,
        site,
        schema_org_data,
        word_count,
        content_markdown: md,
        meta_tags,
    }
}

fn choose_override(override_value: Option<&str>, fallback: String) -> String {
    match override_value.map(str::trim).filter(|s| !s.is_empty()) {
        Some(value) => value.to_string(),
        None => fallback,
    }
}

fn author_hint(parsed: &Html, schema_org_data: Option<&serde_json::Value>) -> (String, String) {
    let author = metadata::extract_author(parsed, schema_org_data, None);
    let published = metadata::extract_published(parsed, schema_org_data, None);
    (author, published)
}

fn apply_removal_step<F>(debug_enabled: bool, step: &str, reason: &str, input: &str, f: F) -> String
where
    F: FnOnce(&str) -> String,
{
    let before_words = dom::count_words(&dom::strip_tags(input));
    let before_chars = input.chars().count();
    let output = f(input);
    let after_words = dom::count_words(&dom::strip_tags(&output));
    let after_chars = output.chars().count();
    observability::removal_step(
        debug_enabled,
        step,
        reason,
        before_words,
        after_words,
        before_chars,
        after_chars,
    );
    output
}

fn apply_llm_markdown_override(result: &mut DistillerResponse, title_hint: &str, markdown: &str) {
    let markdown = markdown::ensure_title_heading(markdown, title_hint);
    let markdown = markdown.trim().to_string();
    if markdown.is_empty() {
        return;
    }

    if let Some(title) = markdown::guess_title(&markdown).filter(|title| !title.trim().is_empty()) {
        result.title = title;
    }

    result.word_count = dom::count_words(&markdown);
    result.description = markdown_description(&markdown);
    result.content = markdown.clone();
    result.content_markdown = Some(markdown);
}

fn markdown_description(markdown: &str) -> String {
    markdown
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

fn recovery_plans(opts: &DistillerOptions) -> Vec<RecoveryPlan> {
    let mut plans = Vec::with_capacity(3);

    let mut partial_selector_retry = opts.clone();
    partial_selector_retry.remove_partial_selectors = false;
    plans.push(RecoveryPlan {
        minimum_word_count: 200,
        options: partial_selector_retry,
        require_double_improvement: true,
    });

    let mut hidden_retry = opts.clone();
    hidden_retry.remove_hidden_elements = false;
    plans.push(RecoveryPlan {
        minimum_word_count: 50,
        options: hidden_retry,
        require_double_improvement: true,
    });

    let mut broad_retry = opts.clone();
    broad_retry.remove_partial_selectors = false;
    broad_retry.remove_content_patterns = false;
    broad_retry.remove_low_scoring = false;
    plans.push(RecoveryPlan {
        minimum_word_count: 50,
        options: broad_retry,
        require_double_improvement: false,
    });

    plans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_override_updates_primary_content_fields() {
        let mut response = DistillerResponse {
            content: "<article><p>heuristic body</p></article>".to_string(),
            content_markdown: Some("# Heuristic Title\n\nheuristic body".to_string()),
            title: "Heuristic Title".to_string(),
            description: "heuristic body".to_string(),
            word_count: 2,
            ..DistillerResponse::default()
        };

        apply_llm_markdown_override(
            &mut response,
            "LLM Title",
            "First llm paragraph.\n\nSecond llm paragraph.",
        );

        assert_eq!(response.title, "LLM Title");
        assert_eq!(
            response.content,
            "# LLM Title\n\nFirst llm paragraph.\n\nSecond llm paragraph."
        );
        assert_eq!(
            response.content_markdown.as_deref(),
            Some("# LLM Title\n\nFirst llm paragraph.\n\nSecond llm paragraph.")
        );
        assert!(response.word_count >= 6);
        assert_eq!(
            response.description,
            "First llm paragraph. Second llm paragraph."
        );
    }

    #[test]
    fn recovery_plans_match_existing_relaxation_order() {
        let plans = recovery_plans(&DistillerOptions::default());

        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].minimum_word_count, 200);
        assert!(!plans[0].options.remove_partial_selectors);
        assert!(plans[0].require_double_improvement);

        assert_eq!(plans[1].minimum_word_count, 50);
        assert!(!plans[1].options.remove_hidden_elements);
        assert!(plans[1].require_double_improvement);

        assert_eq!(plans[2].minimum_word_count, 50);
        assert!(!plans[2].options.remove_partial_selectors);
        assert!(!plans[2].options.remove_content_patterns);
        assert!(!plans[2].options.remove_low_scoring);
        assert!(!plans[2].require_double_improvement);
    }
}
