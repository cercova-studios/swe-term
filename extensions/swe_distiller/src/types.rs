use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParseMode {
    #[default]
    Markdown,
    Html,
    Json,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct MetaTag {
    pub name: Option<String>,
    pub property: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct DistillerResponse {
    pub content: String,
    pub title: String,
    pub description: String,
    pub domain: String,
    pub favicon: String,
    pub image: String,
    pub language: String,
    pub parse_time_ms: u64,
    pub published: String,
    pub author: String,
    pub site: String,
    pub schema_org_data: Option<serde_json::Value>,
    pub word_count: usize,
    pub content_markdown: Option<String>,
    pub meta_tags: Vec<MetaTag>,
}

#[derive(Debug, Clone, Default)]
pub struct DistillerOptions {
    pub debug: bool,
    pub url: Option<String>,
    pub remove_exact_selectors: bool,
    pub remove_partial_selectors: bool,
    pub remove_hidden_elements: bool,
    pub remove_low_scoring: bool,
    pub remove_small_images: bool,
    pub remove_content_patterns: bool,
    pub content_selector: Option<String>,
    pub language: Option<String>,
    pub proxy_url: Option<String>,
    pub mode: ParseMode,
    pub llm: bool,
    pub defaults_applied: bool,
}

impl DistillerOptions {
    pub fn with_defaults(mut self) -> Self {
        self.remove_exact_selectors = true;
        self.remove_partial_selectors = true;
        self.remove_hidden_elements = true;
        self.remove_low_scoring = true;
        self.remove_small_images = true;
        self.remove_content_patterns = true;
        self.defaults_applied = true;
        self
    }
}
