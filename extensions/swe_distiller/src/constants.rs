pub const ENTRY_POINT_SELECTORS: &[&str] = &[
    "#post",
    ".post-content",
    ".post-body",
    ".article-content",
    ".entry-content",
    ".markdown-body",
    "article",
    "[role=\"article\"]",
    "main",
    "[role=\"main\"]",
    ".article-body",
    "#content",
    "body",
];

pub const EXACT_SELECTORS: &[&str] = &[
    "nav",
    "aside",
    "footer",
    "header nav",
    ".sidebar",
    ".advertisement",
    ".ads",
    ".cookie-banner",
    ".social-share",
    ".newsletter",
    ".related-posts",
    "script:not([type^=\"math/\"])",
    "style",
    "noscript",
];

pub const PARTIAL_PATTERNS: &[&str] = &[
    "advert",
    "ads",
    "cookie",
    "newsletter",
    "subscribe",
    "related",
    "social",
    "share",
    "promo",
    "banner",
    "byline",
];

pub const CONTENT_INDICATORS: &[&str] = &[
    "article",
    "content",
    "entry",
    "post",
    "markdown-body",
    "prose",
    "main",
];

pub const NAVIGATION_INDICATORS: &[&str] = &[
    "comments",
    "copyright",
    "subscribe",
    "newsletter",
    "follow us",
    "share",
    "sign in",
    "log in",
    "menu",
];
