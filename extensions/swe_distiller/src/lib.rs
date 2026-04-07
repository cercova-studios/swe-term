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
        let mut applied = false;
        if let Some(md_raw) = llm_markdown {
            let md = markdown::ensure_title_heading(&md_raw, &result.title);
            result.word_count = dom::count_words(&md);
            result.content_markdown = Some(md);
            applied = true;
        }
        observability::debug(
            opts.debug,
            "pipeline.llm",
            format!("outcome={:?}", extraction::llm::outcome_for(applied)),
        );
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

fn parse_html_with_extractor(
    html_str: &str,
    url: Option<&str>,
    mut opts: DistillerOptions,
    extractor_override: Option<ExtractorResult>,
) -> DistillerResponse {
    if !opts.defaults_applied {
        opts = opts.with_defaults();
    }

    let start = Instant::now();
    let mut result = parse_internal(html_str, url, &opts, extractor_override.as_ref());

    if result.word_count < 200 {
        let mut retry_opts = opts.clone();
        retry_opts.remove_partial_selectors = false;
        let retry = parse_internal(html_str, url, &retry_opts, extractor_override.as_ref());
        if retry.word_count > result.word_count.saturating_mul(2) {
            result = retry;
        }
    }

    if result.word_count < 50 {
        let mut retry_opts = opts.clone();
        retry_opts.remove_hidden_elements = false;
        let retry = parse_internal(html_str, url, &retry_opts, extractor_override.as_ref());
        if retry.word_count > result.word_count.saturating_mul(2) {
            result = retry;
        }
    }

    if result.word_count < 50 {
        let mut retry_opts = opts.clone();
        retry_opts.remove_partial_selectors = false;
        retry_opts.remove_content_patterns = false;
        retry_opts.remove_low_scoring = false;
        let retry = parse_internal(html_str, url, &retry_opts, extractor_override.as_ref());
        if retry.word_count > result.word_count {
            result = retry;
        }
    }

    if let Some(schema_text) = metadata::schema_text(result.schema_org_data.as_ref()) {
        let schema_words = dom::count_words(&schema_text);
        if schema_words > (result.word_count * 3) / 2 {
            let mut schema_result = result.clone();
            schema_result.content = schema_text;
            schema_result.word_count = schema_words;
            schema_result.content_markdown =
                Some(markdown::html_to_markdown(&schema_result.content, url));
            result = schema_result;
        }
    }

    result.parse_time_ms = start.elapsed().as_millis() as u64;
    result
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
