import logging
import os
import sys
import base64
import tempfile
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io

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
import openai

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
    faculty = Column(String, default="general")
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
    has_file_upload = Column(Boolean, default=False)
    file_type = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotService:
    def __init__(self, faculty=None):
        """
        Initialize chatbot service with faculty-specific RAG
        
        Args:
            faculty: Faculty for RAG context (if None, will be set per user)
        """
        self.faculty = faculty
        self.embedding_service = None
        if faculty:
            self.embedding_service = EmbeddingService(faculty=faculty)
        
        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            temperature=CHAT_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
            model_name=CHAT_MODEL
        )
        
        # Initialize OpenAI client directly with proper API key
        self.openai_client = openai.OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        # Verify API key is working
        logger.info(f"Initialized OpenAI client with API key: {OPENAI_API_KEY[:20]}...")
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        
        # File memory - store processed files per user session
        self.file_memory = {}  # Format: {user_id: {"file_data": {...}, "filename": "...", "upload_time": datetime}}
    
    def clear_file_memory(self, user_id: int = None):
        """Clear file memory for specific user or all users"""
        if user_id:
            if user_id in self.file_memory:
                del self.file_memory[user_id]
                logger.info(f"Cleared file memory for user {user_id}")
        else:
            self.file_memory.clear()
            logger.info("Cleared all file memory")
    
    def has_uploaded_file(self, user_id: int) -> bool:
        """Check if user has an uploaded file in memory"""
        return user_id in self.file_memory
    
    def get_uploaded_file_info(self, user_id: int) -> Dict:
        """Get uploaded file information for user"""
        if user_id in self.file_memory:
            file_info = self.file_memory[user_id]
            return {
                "filename": file_info["filename"],
                "type": file_info["file_data"]["type"],
                "size": file_info["file_data"].get("size", 0),
                "upload_time": file_info["upload_time"].strftime("%H:%M:%S")
            }
        return None
    
    def _setup_chain_for_faculty(self, faculty: str):
        """Setup the conversational retrieval chain for specific faculty with general fallback"""
        # Create retrievers for both faculty-specific and general collections
        retrievers = []
        
        # Add faculty-specific retriever if not general
        if faculty != "general":
            faculty_embedding_service = EmbeddingService(faculty=faculty)
            faculty_retriever = faculty_embedding_service.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            retrievers.append(("faculty", faculty_retriever, faculty))
        
        # Always add general retriever
        general_embedding_service = EmbeddingService(faculty=None)
        general_retriever = general_embedding_service.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        retrievers.append(("general", general_retriever, "general"))
        
        # Store retrievers for use in chat method
        self.retrievers = retrievers
        
        prompt_template = """
        You are a helpful assistant for the {faculty} faculty. Use the following pieces of context to answer the user's question. 
        The context includes both faculty-specific information and general university information.
        If you don't know the answer based on the context, just say that you don't know politely, don't try to make up an answer.
        Always be polite and professional.

        Context: {{context}}

        Question: {{question}}

        Answer:"""
        
        self.prompt_template = PromptTemplate(
            template=prompt_template.format(faculty=faculty.title()),
            input_variables=["context", "question"]
        )
    
    def _search_multiple_sources(self, query: str) -> tuple:
        """Search both faculty and general sources - 3 from each source"""
        all_docs = []
        sources_info = []
        
        for source_type, retriever, source_faculty in self.retrievers:
            try:
                # Keep original search configuration (3 per source)
                docs = retriever.get_relevant_documents(query)
                for doc in docs:
                    # Add source information to metadata
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source_type'] = source_type
                    doc.metadata['source_faculty'] = source_faculty
                    all_docs.append(doc)
                    
                sources_info.append(f"{len(docs)} from {source_faculty}")
            except Exception as e:
                logger.warning(f"Error searching {source_type} ({source_faculty}): {e}")
        
        return all_docs, sources_info
    
    def _count_tokens_simple(self, text: str) -> int:
        """Simple token estimation: ~4 characters = 1 token"""
        return max(1, len(text) // 4)
    
    def _process_uploaded_file(self, uploaded_file, file_content: bytes) -> Dict:
        """
        Process uploaded file and extract content/images for OpenAI
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_content: Raw file bytes
            
        Returns:
            Dict with processed content and metadata
        """
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                # Image file - encode for vision API
                encoded_image = base64.b64encode(file_content).decode('utf-8')
                return {
                    "type": "image",
                    "content": encoded_image,
                    "filename": uploaded_file.name,
                    "format": file_extension[1:],  # Remove the dot
                    "size": len(file_content)
                }
            
            elif file_extension == '.pdf':
                # PDF file - extract text and images
                return self._process_pdf(file_content, uploaded_file.name)
            
            elif file_extension in ['.txt', '.md']:
                # Text file
                text_content = file_content.decode('utf-8')
                return {
                    "type": "text",
                    "content": text_content,
                    "filename": uploaded_file.name,
                    "size": len(file_content)
                }
            
            else:
                return {
                    "type": "unsupported",
                    "error": f"Unsupported file type: {file_extension}",
                    "filename": uploaded_file.name
                }
                
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            return {
                "type": "error",
                "error": str(e),
                "filename": uploaded_file.name
            }
    
    def _process_pdf(self, pdf_content: bytes, filename: str) -> Dict:
        """Process PDF file to extract text and images"""
        try:
            # Create temporary file with proper cleanup
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_pdf_path = temp_file.name
            
            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(temp_pdf_path)
                
                extracted_text = ""
                images = []
                
                # Process each page
                for page_num in range(len(doc)):
                    try:
                        page = doc[page_num]
                        
                        # Extract text
                        page_text = page.get_text()
                        if page_text.strip():
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        # Extract images (limit to avoid memory issues)
                        if len(images) < 5:  # Limit to first 5 images
                            image_list = page.get_images(full=True)
                            for img_index, img in enumerate(image_list[:2]):  # Max 2 images per page
                                try:
                                    xref = img[0]
                                    pix = fitz.Pixmap(doc, xref)
                                    
                                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                                        img_data = pix.tobytes("png")
                                        encoded_image = base64.b64encode(img_data).decode('utf-8')
                                        images.append({
                                            "page": page_num + 1,
                                            "index": img_index,
                                            "data": encoded_image,
                                            "format": "png"
                                        })
                                    
                                    # Clean up pixmap
                                    pix = None
                                    
                                except Exception as img_error:
                                    logger.warning(f"Could not extract image {img_index} from page {page_num + 1}: {img_error}")
                                    continue
                    
                    except Exception as page_error:
                        logger.warning(f"Error processing page {page_num + 1}: {page_error}")
                        continue
                
                # Close document properly
                page_count = len(doc)
                doc.close()
                
                # Determine primary content type
                if extracted_text.strip() and images:
                    content_type = "pdf_mixed"
                elif extracted_text.strip():
                    content_type = "pdf_text"
                elif images:
                    content_type = "pdf_images"
                else:
                    content_type = "pdf_empty"
                
                return {
                    "type": content_type,
                    "text_content": extracted_text.strip(),
                    "images": images,
                    "filename": filename,
                    "page_count": page_count,
                    "size": len(pdf_content)
                }
                
            finally:
                # Always clean up temporary file
                try:
                    os.unlink(temp_pdf_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            return {
                "type": "error",
                "error": f"Could not process PDF: {str(e)}",
                "filename": filename
            }
    
    def _query_openai_with_file(self, user_query: str, file_data: Dict) -> str:
        """
        Query OpenAI with file content (text or images)
        
        Args:
            user_query: User's question
            file_data: Processed file data
            
        Returns:
            OpenAI response string
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Analyze the provided file and answer the user's question based on its content. Be thorough and helpful."
                }
            ]
            
            # Build user message based on file type
            user_message = {"role": "user", "content": []}
            
            # Add user query
            user_message["content"].append({
                "type": "text",
                "text": f"User question: {user_query}\n\nFile: {file_data['filename']}"
            })
            
            if file_data["type"] == "image":
                # Single image
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{file_data['format']};base64,{file_data['content']}",
                        "detail": "high"  # Use high detail for better analysis
                    }
                })
                
            elif file_data["type"] == "text":
                # Text content
                user_message["content"].append({
                    "type": "text",
                    "text": f"File content:\n{file_data['content']}"
                })
                
            elif file_data["type"] in ["pdf_text", "pdf_mixed"]:
                # PDF with text
                user_message["content"].append({
                    "type": "text",
                    "text": f"PDF content ({file_data['page_count']} pages):\n{file_data['text_content']}"
                })
                
                # Add images if available (limit to first 3 for token management)
                if file_data.get("images"):
                    for img in file_data["images"][:3]:
                        user_message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img['format']};base64,{img['data']}",
                                "detail": "low"  # Use low detail for PDF images to save tokens
                            }
                        })
                        user_message["content"].append({
                            "type": "text",
                            "text": f"^ Image from page {img['page']}"
                        })
                        
            elif file_data["type"] == "pdf_images":
                # PDF with only images
                user_message["content"].append({
                    "type": "text",
                    "text": f"PDF contains {len(file_data['images'])} images from {file_data['page_count']} pages:"
                })
                
                for img in file_data["images"][:4]:  # Limit to first 4 images
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img['format']};base64,{img['data']}",
                            "detail": "high"
                        }
                    })
                    user_message["content"].append({
                        "type": "text",
                        "text": f"^ Page {img['page']}, Image {img['index'] + 1}"
                    })
            
            messages.append(user_message)
            
            # Make API call with proper error handling
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=8000,  # Increased for very long file responses
                temperature=CHAT_TEMPERATURE
            )
            
            # Log token usage for file processing
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"File processing tokens - Total: {response.usage.total_tokens}, "
                           f"Input: {response.usage.prompt_tokens}, "
                           f"Output: {response.usage.completion_tokens}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error querying OpenAI with file: {e}")
            return f"Sorry, I encountered an error processing your file: {str(e)}"
    
    def chat_with_file(self, user_id: int, query: str, uploaded_file, file_content: bytes, db: Session) -> Dict:
        """
        Process chat query with uploaded file - stores file in memory for session
        
        Args:
            user_id: User ID
            query: User's question
            uploaded_file: Streamlit uploaded file object (can be None if using memory)
            file_content: Raw file bytes (can be None if using memory)
            db: Database session
            
        Returns:
            Response dict with file processing results
        """
        try:
            # Check user exists and has tokens
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            if not user.is_active:
                return {"error": "User account is deactivated"}
            
            if user.tokens_used >= user.token_limit:
                return {"error": "Token limit exceeded"}
            
            # Check if we need to process a new file or use memory
            if uploaded_file and file_content:
                # New file uploaded - process and store in memory
                logger.info(f"Processing new file: {uploaded_file.name}")
                file_data = self._process_uploaded_file(uploaded_file, file_content)
                
                if file_data["type"] in ["error", "unsupported"]:
                    return {"error": file_data.get("error", "File processing failed")}
                
                # Store in memory
                self.file_memory[user_id] = {
                    "file_data": file_data,
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now()
                }
                
                logger.info(f"Stored file in memory for user {user_id}: {uploaded_file.name}")
                
            elif user_id in self.file_memory:
                # Use file from memory
                file_info = self.file_memory[user_id]
                file_data = file_info["file_data"]
                logger.info(f"Using file from memory for user {user_id}: {file_info['filename']}")
                
            else:
                return {"error": "No file uploaded or found in memory"}
            
            # Query OpenAI with file content (from memory or newly processed)
            try:
                response = self._query_openai_with_file(query, file_data)
                
                # Calculate tokens using a more accurate method
                user_content_length = len(query)
                
                # Add file content length based on type
                if file_data["type"] == "text":
                    user_content_length += len(file_data["content"])
                elif file_data["type"] in ["pdf_text", "pdf_mixed"]:
                    user_content_length += len(file_data.get("text_content", ""))
                
                # Add base64 image data estimate (images are expensive)
                if file_data["type"] == "image":
                    user_content_length += 1000  # Base estimate for image processing
                elif file_data.get("images"):
                    user_content_length += len(file_data["images"]) * 800  # Estimate per image
                
                response_length = len(response)
                
                # Estimate tokens (more conservative for file uploads)
                estimated_input_tokens = max(user_content_length // 3, 100)  # ~3 chars per token
                estimated_output_tokens = max(response_length // 3, 50)
                total_tokens = estimated_input_tokens + estimated_output_tokens
                
                # Add extra tokens for file processing overhead
                if file_data["type"] in ["image", "pdf_images", "pdf_mixed"]:
                    total_tokens += 500  # Vision API overhead
                
                tokens_used = total_tokens
                
                logger.info(f"File chat token estimate: {tokens_used} (input: {estimated_input_tokens}, output: {estimated_output_tokens})")
                
            except Exception as api_error:
                logger.error(f"Error in file processing API call: {api_error}")
                return {"error": f"File processing failed: {str(api_error)}"}
            
            # Update user token usage
            user.tokens_used += tokens_used
            
            # Get current file info for response
            current_file = self.file_memory[user_id]
            
            # Log the conversation with file info
            chat_log = ChatLog(
                user_id=user_id,
                query=f"[FILE: {current_file['filename']}] {query}",
                response=response,
                tokens_used=tokens_used,
                has_file_upload=True,
                file_type=file_data["type"]
            )
            db.add(chat_log)
            db.commit()
            
            return {
                "response": response,
                "tokens_used": tokens_used,
                "remaining_tokens": user.token_limit - user.tokens_used,
                "file_processed": {
                    "filename": current_file["filename"],
                    "type": file_data["type"],
                    "size": file_data.get("size", 0),
                    "from_memory": uploaded_file is None  # Indicate if loaded from memory
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_file: {e}")
            return {"error": f"An error occurred: {str(e)}"}
    
    def chat_with_memory_file(self, user_id: int, query: str, db: Session) -> Dict:
        """
        Chat using file stored in memory (no new file upload)
        """
        return self.chat_with_file(user_id, query, None, None, db)
    def chat(self, user_id: int, query: str, db: Session) -> Dict:
        """Process chat query - general AI assistant that uses RAG when relevant documents exist"""
        try:
            # Check user exists and has tokens
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            if not user.is_active:
                return {"error": "User account is deactivated"}
            
            if user.tokens_used >= user.token_limit:
                return {"error": "Token limit exceeded"}
            
            # Setup retrievers for user's faculty + general
            user_faculty = user.faculty or "general"
            self._setup_chain_for_faculty(user_faculty)
            
            # Search for relevant documents in both faculty and general collections (3 from each)
            relevant_docs, sources_info = self._search_multiple_sources(query)
            
            # Don't limit documents - use all retrieved chunks for maximum context
            # Total will be ~6 documents (3 faculty + 3 general) with longer chunks
            
            logger.info(f"Retrieved {len(relevant_docs)} documents for query: {query}")
            
            # Log chunk sizes for debugging
            for i, doc in enumerate(relevant_docs):
                chunk_length = len(doc.page_content)
                logger.info(f"Chunk {i+1}: {chunk_length} characters")
            
            # Create enhanced system prompt for general assistant with RAG support
            if relevant_docs:
                # We have relevant documents - use them as primary source
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                logger.info(f"Total context length: {len(context)} characters")
                
                system_prompt = f"""You are an intelligent AI assistant for university students in the {user_faculty.title()} faculty. 

You can answer ANY question students might have - academic, general knowledge, personal advice, technical help, etc.

IMPORTANT: I have found relevant documents from the university's knowledge base that relate to the student's question. Please use this information as your PRIMARY source when answering, but you can also supplement with your general knowledge if needed.

Relevant University Documents:
{context}

Guidelines:
- Provide comprehensive, detailed, and helpful answers using the extensive context provided
- Use the university documents as your main source of truth when they contain relevant information
- If the documents partially answer the question, use them and supplement with your general knowledge
- If the documents don't fully address the question, acknowledge what the documents say and provide additional helpful information
- Always be helpful, accurate, and student-friendly
- Give thorough explanations and examples when appropriate
- Don't hesitate to provide detailed responses - students appreciate comprehensive answers
- Make full use of the extensive context provided to give complete answers"""

                user_prompt = f"Student Question: {query}"
                
            else:
                # No relevant documents - act as general AI assistant
                system_prompt = f"""You are an intelligent AI assistant for university students in the {user_faculty.title()} faculty.

You can help students with ANY questions they might have:
- Academic subjects and coursework
- Study tips and learning strategies  
- Career advice and guidance
- Technical problems and programming
- General knowledge questions
- Personal development and wellness
- University life and student concerns
- Research and writing help
- Math, science, and any other subjects

Provide comprehensive, detailed, and helpful answers. Be thorough in your explanations and give examples when appropriate. Students appreciate detailed responses that fully address their questions."""

                user_prompt = f"Student Question: {query}"
            
            # Create messages for OpenAI API call
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get response with token counting
            try:
                # Make the OpenAI API call
                response = self.openai_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=CHAT_TEMPERATURE,
                    max_tokens=2000  # Increased for more comprehensive answers
                )
                
                response_text = response.choices[0].message.content
                
                # Get actual token usage from OpenAI response
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    
                    # Calculate cost (approximate)
                    # GPT-4o-mini pricing: $0.000015/1K input tokens, $0.00006/1K output tokens
                    input_cost = (prompt_tokens / 1000) * 0.000015
                    output_cost = (completion_tokens / 1000) * 0.00006
                    total_cost = input_cost + output_cost
                    
                    logger.info(f"Token usage - Total: {tokens_used}, Input: {prompt_tokens}, Output: {completion_tokens}, Cost: ${total_cost:.6f}")
                else:
                    # Fallback estimation if usage not available
                    tokens_used = self._count_tokens_simple(query + response_text)
                    prompt_tokens = self._count_tokens_simple(user_prompt)
                    completion_tokens = self._count_tokens_simple(response_text)
                    total_cost = 0.0
                    logger.warning("Using token estimation fallback")
                    
            except Exception as api_error:
                logger.error(f"OpenAI API call failed: {api_error}")
                return {"error": f"AI service error: {str(api_error)}"}
            
            
            # Update user token usage
            user.tokens_used += tokens_used
            
            # Organize source information
            faculty_sources = []
            general_sources = []
            has_rag_sources = len(relevant_docs) > 0
            
            for doc in relevant_docs:
                source_type = doc.metadata.get('source_type', 'unknown')
                content_preview = doc.page_content[:200] + "..."
                
                if source_type == 'faculty':
                    faculty_sources.append(content_preview)
                else:
                    general_sources.append(content_preview)
            
            # Enhanced response - clean, no extra notes
            response_text = response.choices[0].message.content
            
            # Add memory to conversation
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response_text)
            
            # Log the conversation
            chat_log = ChatLog(
                user_id=user_id,
                query=query,
                response=response_text,
                tokens_used=tokens_used,
                has_file_upload=False
            )
            db.add(chat_log)
            db.commit()
            
            return {
                "response": response_text,
                "faculty_sources": faculty_sources,
                "general_sources": general_sources,
                "sources_info": sources_info,
                "has_rag_sources": has_rag_sources,
                "faculty": user_faculty,
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
                    "has_file_upload": log.has_file_upload,
                    "file_type": log.file_type,
                    "timestamp": log.timestamp
                }
                for log in chat_logs
            ]
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def clear_memory(self, user_id: int = None):
        """Clear conversation memory and optionally file memory"""
        self.memory.clear()
        if user_id:
            self.clear_file_memory(user_id)
        logger.info(f"Cleared memory for user {user_id if user_id else 'all'}")
    
    
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
            
            # File upload stats
            file_chats = db.query(ChatLog).filter(ChatLog.has_file_upload == True).count()
            
            return {
                "total_tokens_used": total_tokens,
                "average_tokens_per_chat": round(avg_tokens, 2),
                "total_chats": total_chats,
                "file_upload_chats": file_chats,
                "file_upload_percentage": round((file_chats / total_chats * 100) if total_chats > 0 else 0, 1)
            }
        except Exception as e:
            logger.error(f"Error getting token stats: {e}")
            return {"error": str(e)}
    
    def get_faculty_stats(self, db: Session) -> Dict:
        """Get faculty usage statistics"""
        try:
            from sqlalchemy import func
            
            # Get user counts by faculty
            faculty_users = db.query(
                User.faculty, 
                func.count(User.id).label('user_count')
            ).group_by(User.faculty).all()
            
            # Get token usage by faculty
            faculty_tokens = db.query(
                User.faculty,
                func.sum(User.tokens_used).label('total_tokens')
            ).group_by(User.faculty).all()
            
            # Combine results
            faculty_stats = {}
            for faculty, user_count in faculty_users:
                faculty_stats[faculty or 'general'] = {
                    'users': user_count,
                    'tokens_used': 0
                }
            
            for faculty, token_usage in faculty_tokens:
                faculty_name = faculty or 'general'
                if faculty_name in faculty_stats:
                    faculty_stats[faculty_name]['tokens_used'] = token_usage or 0
            
            return faculty_stats
            
        except Exception as e:
            logger.error(f"Error getting faculty stats: {e}")
            return {"error": str(e)}