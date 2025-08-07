# ===== COMPREHENSIVE CONSTANTS FILE =====
# All constants for the Data Analysis API project

# ===== SYSTEM CONFIGURATION =====

# API Version and basic settings
API_VERSION_CONFIG = "v1"  # From config.py
TIMEOUT = 180  # seconds
MATPLOTLIB_BACKEND = "Agg"  # Non-interactive backend for Docker

# LangChain Model Configuration (from config.py)
# Model configuration is now handled in config.py via get_chat_model()
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
EMBEDDING_MODEL = "text-embedding-ada-002"

# Chain Configuration
CHAIN_TIMEOUT = 120  # seconds
MAX_RETRIES = 3

# Environment Variables (these should be loaded from .env)
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
LANGCHAIN_TRACING_V2_ENV = "LANGCHAIN_TRACING_V2"
LANGCHAIN_API_KEY_ENV = "LANGCHAIN_API_KEY"

# ===== HTTP AND WEB SCRAPING CONSTANTS =====

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

# ===== DATA PROCESSING CONSTANTS =====

# Common data extraction patterns
COMMON_DATA_CLASSES = [
    "chart", "data", "list", "table", "grid", "content", "results"
]

# Numeric cleaning patterns
CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹', '%']  # Added % symbol
SCALE_INDICATORS = [
    'billion', 'million', 'trillion', 'bn', 'mn', 'B', 'M', 'K'
]
FOOTNOTE_PATTERNS = [
    r'\[.*?\]',        # [1], [n 1], etc.
    r'\([^)]*\)',      # Parentheses content
    r'[^\d.\-]'        # Non-numeric except decimal and minus
]

# Text processing constants
WORD_REGEX_PATTERN = r"\b\w+\b"
MIN_KEYWORD_LENGTH = 2

# BeautifulSoup parser
HTML_PARSER = "html.parser"

# Pandas DataFrame operations
PANDAS_ERRORS_COERCE = "coerce"
PANDAS_DTYPE_OBJECT = "object"
PANDAS_REGEX_FALSE = False

# Data cleaning regex patterns (for reference)
FOOTNOTE_REMOVAL_REGEX = r'\\[\\d+\\]'
NON_NUMERIC_REGEX = r'[^\\d.]'

# URL patterns for regex extraction
URL_PATTERN = r"https?://[^\s]+"
S3_PATH_PATTERN = r"s3://[^\s]+"

# JavaScript and JSON patterns
JSON_PATTERN_DOTALL = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
SCRIPT_PATTERN_DOTALL = r'<script[^>]*>(.*?)</script>'
CODE_PATTERN_DOTALL = r'```(?:python)?\n(.*?)\n```'

# File extensions and types
TEXT_FILE_EXTENSIONS = (".txt", ".csv", ".json", ".md")
IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")

# Dictionary keys commonly used
KEY_URL = "url"
KEY_TASK_DESCRIPTION = "task_description"
KEY_FORMAT_ANALYSIS = "format_analysis"
KEY_HTML_CONTENT = "html_content"
KEY_SOUP = "soup"
KEY_JSON_DATA = "json_data"
KEY_FORMAT = "format"
KEY_STRATEGY = "strategy"
KEY_CONFIDENCE = "confidence"
KEY_INCLUDE_VISUALIZATIONS = "include_visualizations"
KEY_VISUALIZATION_FORMAT = "visualization_format"
KEY_MAX_SIZE = "max_size"

# Step numbers and processing
STEP_0_DATA_FORMAT_DETECTION = "Step 0: LLM-powered data format detection"
STEP_1_TABLE_EXTRACTION = "Step 1: Extract tables from webpage"
STEP_2_DATA_CLEANING = "Step 2: Clean and prepare extracted data"
STEP_3_CHART_TYPE_DETECTION = "Step 3: Detect appropriate chart type using LLM"
STEP_4_VISUALIZATION = "Step 4: Generate visualization"
STEP_5_QUESTION_ANSWERING = "Step 5: Answer questions about the data"

# Error messages
ERROR_DATA_FORMAT_DETECTION = "Error in data format detection"
ERROR_TABLE_EXTRACTION = "Error in table extraction"
ERROR_DATA_CLEANING = "Error in data cleaning"
ERROR_CHART_TYPE_DETECTION = "Error in chart type detection"
ERROR_VISUALIZATION = "Error in visualization"
ERROR_QUESTION_ANSWERING = "Error in question answering"

# Print message templates
PRINT_DATA_FORMAT_ANALYSIS = "Data format analysis for {url}:"
PRINT_FORMAT = "Format: {format}"
PRINT_STRATEGY = "Strategy: {strategy}"
PRINT_CONFIDENCE = "Confidence: {confidence}"
PRINT_JSON_DATA_FOUND = "JSON data found: {length} characters"

# Column prefixes and naming
COLUMN_PREFIX = "Column_"

# English stopwords for keyword extraction
ENGLISH_STOPWORDS = {
    "the", "of", "and", "to", "in", "for", "by", "with", "on", "at", "from",
    "as", "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "a", "an", "or", "that", "this", "which", "who", "what", "when", "where",
    "how", "why", "it", "its", "but", "not",
}

# Content selectors for non-table data extraction
CONTENT_SELECTORS = [
    ".chart", ".data-table", ".list", ".grid", ".content",
    "[class*='chart']", "[class*='table']", "[class*='list']",
]

# Valid workflow types for detection
VALID_WORKFLOWS = [
    "data_analysis",
    "image_analysis",
    "code_generation",
    "exploratory_data_analysis",
    "predictive_modeling",
    "data_visualization",
    "web_scraping",
    "multi_step_web_scraping",
    "database_analysis",
    "statistical_analysis",
    "text_analysis",
]

# Workflow name constants
WORKFLOW_MULTI_STEP_WEB_SCRAPING = "multi_step_web_scraping"
WORKFLOW_IMAGE_ANALYSIS = "image_analysis"
WORKFLOW_TEXT_ANALYSIS = "text_analysis"
WORKFLOW_DATA_ANALYSIS = "data_analysis"
WORKFLOW_STATISTICAL_ANALYSIS = "statistical_analysis"
WORKFLOW_DATABASE_ANALYSIS = "database_analysis"
WORKFLOW_DATA_VISUALIZATION = "data_visualization"
WORKFLOW_EXPLORATORY_DATA_ANALYSIS = "exploratory_data_analysis"
WORKFLOW_PREDICTIVE_MODELING = "predictive_modeling"
WORKFLOW_CODE_GENERATION = "code_generation"

# ===== WORKFLOW DETECTION KEYWORDS =====

# Web scraping keywords
SCRAPING_KEYWORDS = [
    "scrape",
    "extract",
    "data from",
    "table from",
    "list of",
    "website",
    "url",
    "html",
    "web page",
    "site",
]

# Multi-step workflow indicators
MULTI_STEP_KEYWORDS = [
    "clean",
    "plot",
    "top",
    "rank",
    "total",
    "answer",
    "question",
    "extract",
    "analyze",
    "visualization",
]

# Image analysis keywords
IMAGE_KEYWORDS = [
    "image",
    "photo",
    "picture",
    "visual",
    "png",
    "jpg",
    "jpeg",
    "computer vision",
    "image processing",
]

# Text analysis keywords
TEXT_KEYWORDS = [
    "text analysis",
    "nlp",
    "sentiment",
    "language",
    "document",
    "natural language processing",
    "text mining",
]

# Legal/Court keywords
LEGAL_KEYWORDS = [
    "court", "judgment", "legal", "case", "disposal", "judge",
    "cnr", "ecourts", "law", "litigation"
]

# Statistical analysis keywords
STATS_KEYWORDS = [
    "correlation",
    "regression",
    "statistical",
    "trend",
    "slope",
    "analysis",
    "statistics",
    "hypothesis",
]

# Database analysis keywords
DB_KEYWORDS = [
    "sql", "duckdb", "database", "query", "parquet",
    "s3://", "mysql", "postgresql", "sqlite"
]

# Data visualization keywords
VIZ_KEYWORDS = [
    "plot",
    "chart",
    "graph",
    "visualization",
    "scatterplot",
    "base64",
    "data uri",
    "histogram",
    "bar chart",
]

# Exploratory data analysis keywords
EDA_KEYWORDS = [
    "explore", "eda", "exploratory", "distribution",
    "summary", "describe", "overview"
]

# Machine learning keywords
ML_KEYWORDS = [
    "predict",
    "model",
    "machine learning",
    "ml",
    "forecast",
    "classification",
    "clustering",
    "neural network",
]

# Code generation keywords
CODE_KEYWORDS = [
    "generate code", "python code", "script", "function", "programming", "code"
]

# Generic web keywords
WEB_KEYWORDS = ["scrape", "extract", "web", "html", "website"]

# Additional hardcoded keyword lists for detection
PLOT_CHART_KEYWORDS = ["plot", "chart", "graph", "visualization", "base64", "data uri"]
FORMAT_KEYWORDS = ["json", "csv", "table"]
FINANCIAL_DETECTION_KEYWORDS = ["gross", "revenue", "profit"]
RANKING_DETECTION_KEYWORDS = ["rank", "position", "top"]
DATABASE_DETECTION_KEYWORDS = ["duckdb", "sql", "database"]
CHART_TYPE_KEYWORDS = ["scatterplot", "scatter"]
REGRESSION_KEYWORDS = ["regression"]
BASE64_KEYWORDS = ["base64", "data uri"]

# ===== DATA ANALYSIS CONSTANTS =====

# Summary filtering keywords
SUMMARY_KEYWORDS = [
    "world",
    "total",
    "sum",
    "all",
    "global",
    "aggregate",
    "overall",
    "average",
    "mean",
    "median",
    "other",
    "others",
]

# Financial data indicators
FINANCIAL_KEYWORDS = [
    "gross",
    "revenue",
    "gdp",
    "billion",
    "million",
    "box office",
    "earnings",
]

# Health data indicators
HEALTH_KEYWORDS = [
    "cases",
    "deaths",
    "covid",
    "infection",
    "recovery",
    "mortality",
    "disease",
]

# Sports data indicators
SPORTS_KEYWORDS = [
    "runs",
    "average",
    "cricket",
    "batsmen",
    "matches",
    "innings",
    "wickets",
    "goals",
    "points",
]

# Economic data indicators
ECONOMIC_KEYWORDS = [
    "inflation",
    "cpi",
    "rate",
    "economics",
    "trading",
    "price",
    "index",
]

# Entertainment data indicators
ENTERTAINMENT_KEYWORDS = [
    "rating",
    "imdb",
    "score",
    "movie",
    "film",
    "review",
]

# ===== CHART AND VISUALIZATION CONSTANTS =====

# Chart colors and styling
CHART_COLORS = {
    "primary": "skyblue",
    "secondary": "lightgreen",
    "accent": "red",
    "warning": "orange",
    "error": "darkred",
}

# Chart styling parameters (consolidated from above)
CHART_STYLE = {
    "alpha": 0.7,
    "alpha_scatter": 0.6,
    "alpha_regression": 0.8,
    "marker_size": 50,
    "marker_size_small": 6,
    "line_width": 2,
    "rotation": 45,
    "ha": "right",
}

# Chart type constants
CHART_TYPE_BAR = "bar"
CHART_TYPE_SCATTER = "scatter"
CHART_TYPE_HISTOGRAM = "histogram"
CHART_TYPE_TIME_SERIES = "time_series"

# Visualization formats
VISUALIZATION_FORMAT_BASE64 = "base64_data_uri"
# Alias for output format
OUTPUT_FORMAT_BASE64 = VISUALIZATION_FORMAT_BASE64

# ===== API CONSTANTS =====

# API metadata
API_TITLE = "Data Analysis API"
API_DESCRIPTION = "An API that uses LLMs to source, prepare, analyze, and visualize any data with multi-modal support."
API_VERSION = "2.0.0"

# API features
API_FEATURES = [
    "Multiple file upload support",
    "Synchronous processing",
    "LLM-based workflow detection",
    "Multi-modal analysis (text, image, code)",
    "12+ specialized workflows",
]

# API endpoints
API_ENDPOINTS = {
    "main": "/api/ (POST - multiple files with required questions.txt)",
    "health": "/health (GET)",
    "docs": "/docs (GET - Swagger UI)",
}

# Default values
DEFAULT_WORKFLOW = "data_analysis"
DEFAULT_PRIORITY = "normal"
DEFAULT_TARGET_AUDIENCE = "technical team"
DEFAULT_PIPELINE_TYPE = "custom"

# Output requirements
DEFAULT_OUTPUT_REQUIREMENTS = {
    "format": "json",
    "include_visualizations": False,
    "response_time_limit": "3 minutes"
}

# ===== NUMERIC CONSTANTS =====

# File processing limits
MAX_FILE_SIZE = 100000  # 100KB limit for base64 images
TIMEOUT_SECONDS = 180  # 3 minutes
MAX_SIZE_BYTES = "100000 bytes"  # String version for output requirements

# Encoding
ENCODING = "utf-8"

# Table processing parameters
TABLE_HEAD_ROWS = 3
TABLE_MAX_COLS = 10
TABLE_MAX_ROWS = 3
TABLE_MIN_ROWS = 10
TABLE_MIN_COLS = 2
SEPARATOR_LINE = "-" * 50

# Chart styling parameters
CHART_ALPHA = 0.7
CHART_ALPHA_SCATTER = 0.6
CHART_ALPHA_REGRESSION = 0.8
CHART_MARKER_SIZE = 50
CHART_MARKER_SIZE_SMALL = 6
CHART_LINE_WIDTH = 2
CHART_ROTATION = 45
CHART_HORIZONTAL_ALIGNMENT = "right"

# HTTP Status Codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_TIMEOUT = 408
HTTP_INTERNAL_ERROR = 500

# Container and deployment constants
CONTAINER_IMAGE_NAME = "data-analysis-api"
CONTAINER_NAME = "data-analysis-api-container"
CONTAINER_PORT = 8000

# System paths (chain imports)
CHAINS_DIRECTORY = "chains"

# File size and character limits
JSON_DATA_CHAR_DISPLAY_LIMIT = 1000  # For printing JSON data info
TASK_DESCRIPTION_DISPLAY_LIMIT = 200  # For logging task description

# Default number values used in processing
DEFAULT_ZERO = 0
DEFAULT_ONE = 1
DEFAULT_TWO = 2
DEFAULT_THREE = 3
FALLBACK_TABLE_COUNT = 1

# Status messages
STATUS_OPERATIONAL = "operational"
STATUS_HEALTHY = "healthy"
STATUS_AVAILABLE = "available"
STATUS_UNAVAILABLE = "unavailable"

# Logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "app.log"

# Static files
STATIC_DIRECTORY = "."
STATIC_NAME = "static"

# Data format strategies
STRATEGY_JSON_PARSING = "json_parsing"
STRATEGY_REGEX_EXTRACTION = "regex_extraction"
STRATEGY_CUSTOM_PARSING = "custom_parsing"
STRATEGY_PANDAS_READ_HTML = "pandas_read_html"

# Format types
FORMAT_HTML_TABLES = "html_tables"
FORMAT_JSON = "json"

# Confidence levels
CONFIDENCE_LOW = "low"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_HIGH = "high"

# Script types
SCRIPT_TYPE_JSON_LD = "application/ld+json"

# ===== DOMAIN TYPE CONSTANTS =====
DATA_TYPE_FINANCIAL = "financial"
DATA_TYPE_HEALTH = "health"
DATA_TYPE_SPORTS = "sports"
DATA_TYPE_ECONOMIC = "economic"
DATA_TYPE_ENTERTAINMENT = "entertainment"
DATA_TYPE_OTHER = "other"
DATA_TYPE_RANKING = "ranking"

# ===== DATABASE & FORMAT CONSTANTS =====
DATABASE_TYPE_SQL = "sql"
FILE_FORMAT_PARQUET = "parquet"

# ===== CONTENT TYPE CONSTANTS =====
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_TEXT = "text/plain"
