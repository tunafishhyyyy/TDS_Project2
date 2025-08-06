# Standard HTTP headers and constants for web scraping

# Standard User-Agent for web requests
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)

# Standard request headers
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

# Common data extraction patterns
COMMON_DATA_CLASSES = [
    "chart", "data", "list", "table", "grid", "content", "results"
]

# Numeric cleaning patterns
CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹']
SCALE_INDICATORS = ['billion', 'million', 'trillion', 'bn', 'mn', 'B', 'M', 'K']
FOOTNOTE_PATTERNS = [
    r'\[.*?\]',        # [1], [n 1], etc.
    r'\([^)]*\)',      # Parentheses content
    r'[^\d.\-]'        # Non-numeric except decimal and minus
]
