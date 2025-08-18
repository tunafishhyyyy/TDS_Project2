# Configuration settings for the project
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_VERSION = "v1"
TIMEOUT = 180  # seconds

# LangChain Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Added for Gemini

# Enhanced LLM Configuration - Multiple Gemini keys for fallback
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k] or [GEMINI_API_KEY]

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Model Configuration with hierarchy
DEFAULT_OPENAI_MODEL = "gpt-4"  # Upgraded to GPT-4 for better accuracy
DEFAULT_GEMINI_MODEL = "gemini-pro"

# Enhanced model hierarchy for fallback
GEMINI_MODEL_HIERARCHY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite", 
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

TEMPERATURE = 0.3  # Lower temperature for more accurate analysis
MAX_TOKENS = 4000

# Enhanced reliability settings
USE_ENHANCED_LLM = os.getenv("USE_ENHANCED_LLM", "true").lower() == "true"
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "240"))

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
EMBEDDING_MODEL = "text-embedding-ada-002"  # Reverted for stability

# Chain Configuration
CHAIN_TIMEOUT = 120  # seconds
MAX_RETRIES = 3


def get_chat_model(provider="openai"):
    """
    Get a chat model instance with configuration based on the provider.
    :param provider: 'openai' or 'gemini'
    """
    if provider == "openai":
        try:
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model_name=DEFAULT_OPENAI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        except ImportError:
            # Fallback for newer LangChain versions
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model=DEFAULT_OPENAI_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
            except ImportError:
                raise ImportError("Could not import ChatOpenAI from langchain or langchain_openai")
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=DEFAULT_GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
                convert_system_message_to_human=True  # Important for some chains
            )
        except ImportError:
            raise ImportError("Could not import ChatGoogleGenerativeAI. Please install langchain-google-genai.")
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'gemini'.")
