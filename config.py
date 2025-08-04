import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
    print("❌ ERROR: OpenAI API key not found!")
    print("Please check your .env file and make sure OPENAI_API_KEY is set correctly.")
    print("Current API key value:", OPENAI_API_KEY[:20] + "..." if OPENAI_API_KEY else "None")
    raise ValueError("OpenAI API key is required")

if not OPENAI_API_KEY.startswith("sk-"):
    print("❌ ERROR: Invalid OpenAI API key format!")
    print("API key should start with 'sk-'")
    print("Current API key starts with:", OPENAI_API_KEY[:10] if OPENAI_API_KEY else "None")
    raise ValueError("Invalid OpenAI API key format")

print(f"✅ OpenAI API key loaded successfully: {OPENAI_API_KEY[:20]}...")

# Chroma Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "company_documents"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

# Admin Configuration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Token Limits
DEFAULT_TOKEN_LIMIT = int(os.getenv("DEFAULT_TOKEN_LIMIT", "1000"))
MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "10000"))

# Faculty Configuration
FACULTIES = ["engineering", "business", "science", "arts"]

# Text Chunking Configuration - Admin fully configurable
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))  # Admin configurable default
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))  # Admin configurable default
MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "0"))  # 0 = no limit

# Retrieval Configuration - Simple defaults
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "3"))  # 3 chunks as requested
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Chat Configuration
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "10"))

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Performance Settings - Standard settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10")) 
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# File Processing Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.csv']