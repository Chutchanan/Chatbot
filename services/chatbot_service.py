import logging
import os
import sys
from typing import List, Optional, Dict
from datetime import datetime

# Add current directory to path to fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks import get_openai_callback
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base

# Import embedding service
from services.embedding_service import EmbeddingService

# Import config
try:
    from config import OPENAI_API_KEY, CHAT_MODEL, CHAT_TEMPERATURE, DATABASE_URL
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    CHAT_MODEL = "gpt-4o-mini"
    CHAT_TEMPERATURE = 0.7
    DATABASE_URL = "sqlite:///./chatbot.db"

# Define models directly here to avoid import issues
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    token_limit = Column(Integer, default=1000)
    tokens_used = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatLog(Base):
    __tablename__ = "chat_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    query = Column(Text)
    response = Column(Text)
    tokens_used = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm = ChatOpenAI(
            temperature=CHAT_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
            model_name=CHAT_MODEL
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the conversational retrieval chain"""
        prompt_template = """
        You are a helpful assistant for Bluebik Vulcan. Use the following pieces of context to answer the user's question. 
        If you don't know the answer based on the context, just say that you don't know politely, don't try to make up an answer.
        Always be polite and professional.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.embedding_service.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            output_key="answer"
        )
    
    def _count_tokens_simple(self, text: str) -> int:
        """Simple token estimation: ~4 characters = 1 token"""
        return max(1, len(text) // 4)
    
    def chat(self, user_id: int, query: str, db: Session) -> Dict:
        """Process chat query and return response with accurate token counting"""
        try:
            # Check user exists and has tokens
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            if not user.is_active:
                return {"error": "User account is deactivated"}
            
            if user.tokens_used >= user.token_limit:
                return {"error": "Token limit exceeded"}
            
            # Get response with accurate token counting using OpenAI callback
            try:
                with get_openai_callback() as cb:
                    result = self.qa_chain({"question": query})
                    
                    # Use actual token count from OpenAI API
                    tokens_used = cb.total_tokens
                    prompt_tokens = cb.prompt_tokens
                    completion_tokens = cb.completion_tokens
                    total_cost = cb.total_cost
                    
            except Exception as callback_error:
                logger.warning(f"OpenAI callback failed, using simple estimation: {callback_error}")
                # Fallback to simple estimation
                result = self.qa_chain({"question": query})
                tokens_used = self._count_tokens_simple(query) + self._count_tokens_simple(result["answer"])
                prompt_tokens = self._count_tokens_simple(query)
                completion_tokens = self._count_tokens_simple(result["answer"])
                total_cost = 0.0
            
            response = result["answer"]
            source_documents = result.get("source_documents", [])
            
            # Update user token usage
            user.tokens_used += tokens_used
            
            # Log the conversation
            chat_log = ChatLog(
                user_id=user_id,
                query=query,
                response=response,
                tokens_used=tokens_used
            )
            db.add(chat_log)
            db.commit()
            
            return {
                "response": response,
                "sources": [doc.page_content[:200] + "..." for doc in source_documents],
                "tokens_used": tokens_used,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(total_cost, 6),
                "remaining_tokens": user.token_limit - user.tokens_used
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {"error": f"An error occurred: {str(e)}"}
    
    def get_chat_history(self, user_id: int, db: Session, limit: int = 10) -> List[Dict]:
        """Get chat history for a user"""
        try:
            chat_logs = db.query(ChatLog).filter(
                ChatLog.user_id == user_id
            ).order_by(ChatLog.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    "id": log.id,
                    "query": log.query,
                    "response": log.response,
                    "tokens_used": log.tokens_used,
                    "timestamp": log.timestamp
                }
                for log in chat_logs
            ]
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def get_token_stats(self, db: Session) -> Dict:
        """Get token usage statistics"""
        try:
            from sqlalchemy import func
            
            # Total tokens used across all users
            total_tokens = db.query(func.sum(ChatLog.tokens_used)).scalar() or 0
            
            # Average tokens per chat
            avg_tokens = db.query(func.avg(ChatLog.tokens_used)).scalar() or 0
            
            # Total chats
            total_chats = db.query(ChatLog).count()
            
            return {
                "total_tokens_used": total_tokens,
                "average_tokens_per_chat": round(avg_tokens, 2),
                "total_chats": total_chats
            }
        except Exception as e:
            logger.error(f"Error getting token stats: {e}")
            return {"error": str(e)}