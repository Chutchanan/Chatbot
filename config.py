import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Chroma Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "company_documents"

# Database Configuration
DATABASE_URL = "sqlite:///./chatbot.db"

# Admin Configuration
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Token Limits
DEFAULT_TOKEN_LIMIT = 1000
MAX_TOKEN_LIMIT = 10000

# Text Chunking Configuration - Just simple numbers
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))           # Size of each text chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))      # Overlap between chunks
MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "1000"))  # Limit chunks per doc

# Retrieval Configuration
SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_K", "3"))  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))  # Minimum similarity score

# Chat Configuration
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))    # LLM creativity (0.0-1.0)
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "10"))       # Number of previous messages to remember

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")  # OpenAI embedding model
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")  # OpenAI chat model

# Performance Settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # Number of documents to process at once
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Retry attempts for API calls

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR

# File Processing Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))  # Maximum file size in MB
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.csv']  # Supported file types

# Simple chunk size update function
def update_chunk_config(chunk_size=None, chunk_overlap=None):
    """Update chunk configuration at runtime"""
    global CHUNK_SIZE, CHUNK_OVERLAP
    if chunk_size is not None:
        CHUNK_SIZE = chunk_size
    if chunk_overlap is not None:
        CHUNK_OVERLAP = chunk_overlap
    print(f"Updated chunk config: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

# Configuration validation
def validate_config():
    """Validate configuration values"""
    issues = []
    
    if OPENAI_API_KEY == "your-api-key-here" or not OPENAI_API_KEY:
        issues.append("OpenAI API key is not set")
    
    if CHUNK_SIZE < 100:
        issues.append("CHUNK_SIZE should be at least 100")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        issues.append("CHUNK_OVERLAP should be less than CHUNK_SIZE")
    
    if SIMILARITY_SEARCH_K < 1:
        issues.append("SIMILARITY_SEARCH_K should be at least 1")
    
    return issues

# Print configuration summary
def print_config_summary():
    """Print current configuration"""
    print("🔧 CURRENT CONFIGURATION")
    print("=" * 30)
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"Search Results (k): {SIMILARITY_SEARCH_K}")
    print(f"Chat Model: {CHAT_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Temperature: {CHAT_TEMPERATURE}")
    print(f"Max Chat History: {MAX_CHAT_HISTORY}")
    
    issues = validate_config()
    if issues:
        print("\n⚠️ Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ Configuration looks good!")

if __name__ == "__main__":
    print_config_summary()