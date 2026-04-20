use std::fs;
use std::io::{self, Write};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use swe_distiller::types::{DistillerOptions, ParseMode};

#[derive(Debug, Clone, ValueEnum)]
enum OutputMode {
    Markdown,
    Json,
}

#[derive(Debug, Parser)]
#[command(
    name = "swe_distiller",
    about = "Extract article content from web pages"
)]
struct Cli {
    /// URL to extract
    url: String,

    /// Output file (default: webpage.md). Use "-" or omit with --stdout to print to stdout.
    #[arg(short, long)]
    output: Option<String>,

    /// Write output to stdout instead of a file (for piping to other tools)
    #[arg(long)]
    stdout: bool,

    /// Output mode
    #[arg(long, value_enum, default_value = "markdown")]
    mode: OutputMode,

    /// Preferred language (BCP 47)
    #[arg(short, long)]
    lang: Option<String>,

    /// Optional outbound HTTP/HTTPS proxy URL (for example: http://user:pass@host:port)
    #[arg(long)]
    proxy: Option<String>,

    /// Enable debug behavior
    #[arg(long)]
    debug: bool,

    /// Enable LLM extraction pipeline
    #[arg(long)]
    llm: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    let mode = match args.mode {
        OutputMode::Markdown => ParseMode::Markdown,
        OutputMode::Json => ParseMode::Json,
    };

    let opts = DistillerOptions {
        language: args.lang,
        proxy_url: args.proxy,
        debug: args.debug,
        llm: args.llm,
        mode,
        ..DistillerOptions::default()
    };

    let result = swe_distiller::extract_url(&args.url, opts).await?;

    if result.word_count == 0 {
        anyhow::bail!("No content could be extracted");
    }

    let output = match mode {
        ParseMode::Markdown => result.content_markdown.clone().unwrap_or_default(),
        ParseMode::Json => serde_json::to_string_pretty(&result)?,
    };

    let use_stdout = args.stdout || args.output.as_deref() == Some("-");

    if use_stdout {
        io::stdout()
            .write_all(output.as_bytes())
            .context("Failed to write to stdout")?;
    } else {
        let target = if let Some(path) = args.output {
            path
        } else {
            match mode {
                ParseMode::Markdown => "webpage.md".to_string(),
                ParseMode::Json => "webpage.json".to_string(),
            }
        };

        fs::write(&target, &output).with_context(|| format!("Failed to write {target}"))?;
        eprintln!("Wrote output to {target}");
    }

    Ok(())
}
