# Configuration settings for the project
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_VERSION = "v1"
TIMEOUT = 180  # seconds

# LangChain Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Model Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
EMBEDDING_MODEL = "text-embedding-ada-002"

# Chain Configuration
CHAIN_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
