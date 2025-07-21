# services/__init__.py
# This file makes the services directory a Python package

# Import services for easier access
try:
    from .embedding_service import EmbeddingService
    from .chatbot_service import ChatbotService
except ImportError:
    # If imports fail, they'll be imported when needed
    pass