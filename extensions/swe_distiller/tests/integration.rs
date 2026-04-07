use swe_distiller::{parse_html, types::DistillerOptions};

#[test]
fn extracts_basic_article_markdown() {
    let html = r#"
    <html>
      <head><title>Test Page</title></head>
      <body>
        <nav>Menu</nav>
        <article>
          <h1>Test Page</h1>
          <p>Hello world from swe_distiller.</p>
          <p>This should be preserved as content.</p>
        </article>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    "#;

    let result = parse_html(
        html,
        Some("https://example.com/post"),
        DistillerOptions::default(),
    );
    let md = result.content_markdown.unwrap_or_default();

    assert!(result.word_count > 5);
    assert!(md.contains("Hello world"));
    assert!(!md.to_lowercase().contains("menu"));
}

#[test]
fn fixture_smoke_produces_non_empty_output() {
    let html = include_str!("../tests/fixtures/general--cp4space-jordan-algebra.html");
    let result = parse_html(
        html,
        Some("https://cp4space.hatsya.com/example"),
        DistillerOptions::default(),
    );

    assert!(result.word_count > 100);
    assert!(!result.content.is_empty());
    assert!(result.content_markdown.as_deref().unwrap_or_default().len() > 100);
}

#[test]
fn trims_mintlify_style_chrome_and_related_blocks() {
    let html = r#"
    <html>
      <head><title>How we built a virtual filesystem for our Assistant</title></head>
      <body>
        <header>
          <a href="/contact/sales">Contact sales</a>
          <a href="/signup">Start for free</a>
        </header>
        <article>
          <h1>How we built a virtual filesystem for our Assistant</h1>
          <p>March 24, 2026</p>
          <p>Dens Sumesh</p>
          <p>RAG is great, until it isn't.</p>
          <h2>The Container Bottleneck</h2>
          <p>Most harnesses solve this by spinning up a sandbox.</p>
          <h2>Conclusion</h2>
          <p>By replacing sandboxes with a virtual filesystem, we got faster boot.</p>
          <h3>More blog posts to read</h3>
          <p>Docs on autopilot</p>
          <p>Share this article</p>
        </article>
        <footer>
          <h3>explore</h3>
        </footer>
      </body>
    </html>
    "#;

    let result = parse_html(
        html,
        Some("https://www.mintlify.com/blog/how-we-built-a-virtual-filesystem-for-our-assistant"),
        DistillerOptions::default(),
    );

    let md = result.content_markdown.unwrap_or_default().to_lowercase();
    assert!(md.contains("how we built a virtual filesystem for our assistant"));
    assert!(md.contains("the container bottleneck"));
    assert!(md.contains("conclusion"));

    assert!(!md.contains("contact sales"));
    assert!(!md.contains("start for free"));
    assert!(!md.contains("more blog posts to read"));
    assert!(!md.contains("docs on autopilot"));
    assert!(!md.contains("share this article"));
    assert!(!md.contains("explore"));
}

#[test]
fn injects_title_heading_when_article_body_lacks_h1() {
    let html = r#"
    <html>
      <head>
        <title>Claude Code Source Leak Analysis</title>
        <meta property="og:title" content="Claude Code Source Leak Analysis" />
      </head>
      <body>
        <article>
          <p><strong>Update:</strong> see HN discussions.</p>
          <p>This post starts with an update paragraph and no visible h1 in the extracted body.</p>
          <h2>Key Findings</h2>
          <p>A lot of details followed from source analysis.</p>
        </article>
      </body>
    </html>
    "#;

    let result = parse_html(
        html,
        Some("https://example.com/posts/claude-code-source-leak"),
        DistillerOptions::default(),
    );
    let md = result.content_markdown.unwrap_or_default();

    assert!(md.starts_with("# Claude Code Source Leak Analysis"));
    assert!(md.to_lowercase().contains("update:"));
    assert!(md.contains("## Key Findings"));
}

#[test]
fn keeps_paragraph_breaks_between_adjacent_paragraph_tags() {
    let html = r#"
    <html>
      <head><title>Paragraph Break Test</title></head>
      <body>
        <article>
          <h1>Paragraph Break Test</h1>
          <p><strong>Update:</strong> see HN discussions about this post: <a href="https://news.ycombinator.com/item?id=1">https://news.ycombinator.com/item?id=1</a></p>
          <p>I use Claude Code daily, so I wanted to inspect this package.</p>
        </article>
      </body>
    </html>
    "#;

    let result = parse_html(
        html,
        Some("https://example.com/post"),
        DistillerOptions::default(),
    );
    let md = result.content_markdown.unwrap_or_default();

    assert!(md.contains("item?id=1)\n\nI use Claude Code daily"));
}

#[test]
fn preserves_breaks_between_byline_date_and_body_in_block_wrappers() {
    let html = r#"
    <html>
      <head><title>How Microsoft Vaporized a Trillion Dollars</title></head>
      <body>
        <article>
          <h2>How Microsoft Vaporized a Trillion Dollars</h2>
          <h3>Inside the complacency and decisions that eroded trust in Azure.</h3>
          <div class="byline">
            <div><a href="https://substack.com/@isolveproblems">Axel Rietschin</a></div>
            <div>Mar 29, 2026</div>
          </div>
          <div>
            <p>This is the first of a series of articles in which you will learn what happened.</p>
          </div>
        </article>
      </body>
    </html>
    "#;

    let result = parse_html(
        html,
        Some("https://isolveproblems.substack.com/p/how-microsoft-vaporized-a-trillion"),
        DistillerOptions::default(),
    );
    let md = result.content_markdown.unwrap_or_default();

    assert!(md.contains("[Axel Rietschin](https://substack.com/@isolveproblems)"));
    assert!(md.contains("Mar 29, 2026"));
    let date_idx = md.find("Mar 29, 2026").unwrap_or(usize::MAX);
    let body_idx = md
        .find("This is the first of a series")
        .unwrap_or(usize::MAX);
    assert!(date_idx < body_idx);
    assert!(!md.contains("Mar 29, 2026This is the first of a series"));
}

#[test]
fn uses_substack_preload_extractor_for_body_and_metadata() {
    let html = r#"
    <!doctype html>
    <html>
      <head>
        <meta property="og:title" content="Fallback Title" />
      </head>
      <body>
        <script>
          window._preloads = JSON.parse("{\"feedData\":{\"initialPost\":{\"post\":{\"title\":\"How Microsoft Vaporized a Trillion-Dollar Bet\",\"subtitle\":\"A Substack post\",\"body_html\":\"<p>This is the first paragraph.</p><p>Second paragraph with details.</p>\",\"post_date\":\"2026-03-29T00:00:00Z\",\"publishedBylines\":[{\"name\":\"Axel Rietschin\"}]}}}}");
        </script>
      </body>
    </html>
    "#;

    let resp = parse_html(
        html,
        Some("https://isolveproblems.substack.com/p/how-microsoft-vaporized-a-trillion"),
        DistillerOptions::default(),
    );
    let md = resp
        .content_markdown
        .expect("markdown should be generated from extracted substack body");

    assert!(md.contains("# How Microsoft Vaporized a Trillion-Dollar Bet"));
    assert!(md.contains("This is the first paragraph."));
    assert!(md.contains("Second paragraph with details."));
    assert_eq!(resp.author, "Axel Rietschin");
    assert_eq!(resp.site, "Substack");
    assert_eq!(resp.published, "2026-03-29T00:00:00Z");
}

#[test]
fn uses_github_extractor_on_markdown_body() {
    let html = r#"
    <html><body>
      <article class="markdown-body">
        <h1>Project Title</h1>
        <p>This README includes substantial usage documentation and setup details.</p>
      </article>
    </body></html>
    "#;
    let resp = parse_html(
        html,
        Some("https://github.com/acme/project"),
        DistillerOptions::default(),
    );
    let md = resp.content_markdown.unwrap_or_default();
    assert!(md.contains("Project Title"));
    assert!(md.contains("substantial usage documentation"));
}

#[test]
fn uses_x_extractor_for_tweet_text() {
    let html = r#"
    <html><body>
      <article>
        <div data-testid="tweetText">This is a long-form tweet thread opener with enough words to pass extraction quality.</div>
      </article>
    </body></html>
    "#;
    let resp = parse_html(
        html,
        Some("https://x.com/someone/status/123"),
        DistillerOptions::default(),
    );
    let md = resp.content_markdown.unwrap_or_default();
    assert!(md.contains("long-form tweet thread opener"));
}
