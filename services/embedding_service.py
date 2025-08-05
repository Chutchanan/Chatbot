import os
import logging
from typing import List, Optional, Dict
from pathlib import Path
import chromadb
from langchain.text_splitter import TextSplitter

# FIXED: Updated imports to avoid deprecation warnings
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Add these imports for config
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from config import (
        OPENAI_API_KEY, 
        CHROMA_PERSIST_DIRECTORY, 
        COLLECTION_NAME,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        MAX_CHUNKS_PER_DOCUMENT,
        SIMILARITY_SEARCH_K,
        EMBEDDING_MODEL,
        BATCH_SIZE,
        MAX_RETRIES,
        LOG_LEVEL,
        MAX_FILE_SIZE_MB,
        SUPPORTED_EXTENSIONS,
        FACULTIES,
        validate_chunk_settings,
        MIN_CHUNK_SIZE,
        MAX_CHUNK_SIZE
    )
except ImportError:
    # Fallback config if import fails
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "company_documents"
    DEFAULT_CHUNK_SIZE = 5000
    DEFAULT_CHUNK_OVERLAP = 500
    MAX_CHUNKS_PER_DOCUMENT = 1000
    SIMILARITY_SEARCH_K = 3
    EMBEDDING_MODEL = "text-embedding-ada-002"
    BATCH_SIZE = 10
    MAX_RETRIES = 3
    LOG_LEVEL = "INFO"
    MAX_FILE_SIZE_MB = 10
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.csv']
    FACULTIES = ["engineering", "business", "science", "arts"]
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 50000
    
    def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> tuple:
        chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, chunk_size))
        max_overlap = min(chunk_size // 2, 15000)
        chunk_overlap = max(0, min(max_overlap, chunk_overlap))
        return chunk_size, chunk_overlap

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

class PureSizeTextSplitter(TextSplitter):
    """
    FIXED: Custom text splitter that ONLY uses chunk size - NO separators at all
    This ensures chunks are exactly the size specified by admin with proper overlap handling
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based PURELY on character count - no separators"""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # Ensure overlap is not larger than chunk size
        effective_overlap = min(self.chunk_overlap, self.chunk_size // 2)
        
        logger.info(f"🔤 SPLITTING TEXT: {text_length:,} chars into {self.chunk_size:,} char chunks with {effective_overlap:,} overlap")
        
        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + self.chunk_size, text_length)
            
            # Extract chunk
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
                logger.debug(f"Chunk {len(chunks)}: start={start}, end={end}, length={len(chunk)}")
            
            # Calculate next start position
            # Move forward by (chunk_size - overlap) to get proper overlap
            next_start = start + self.chunk_size - effective_overlap
            
            # Ensure we're making progress
            if next_start <= start:
                next_start = start + 1
            
            start = next_start
            
            # Break if we've reached or passed the end
            if start >= text_length:
                break
        
        logger.info(f"✅ CHUNKING COMPLETE: Created {len(chunks)} chunks from {text_length:,} characters")
        
        # Log first few chunk sizes for verification
        if chunks:
            sample_sizes = [len(chunk) for chunk in chunks[:5]]
            logger.info(f"📊 First 5 chunk sizes: {sample_sizes}")
            avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            logger.info(f"📊 Average chunk size: {avg_size:.0f} characters")
        
        return chunks

class EmbeddingService:
    def __init__(self, faculty=None, custom_chunk_size=None, custom_chunk_overlap=None):
        """
        Initialize embedding service with faculty support and ADMIN DYNAMIC CHUNK SETTINGS
        NOW USES PURE SIZE-ONLY SPLITTING (no separators at all)
        
        Args:
            faculty: Faculty name for faculty-specific collections
            custom_chunk_size: Admin-specified chunk size (overrides defaults completely)
            custom_chunk_overlap: Admin-specified chunk overlap (overrides defaults completely)
        """
        # Set faculty and collection name
        self.faculty = faculty
        self.collection_name = self._get_collection_name(faculty)
        
        # ADMIN DYNAMIC CHUNK CONFIGURATION - Full control to admin
        if custom_chunk_size is not None or custom_chunk_overlap is not None:
            # Admin specified custom settings - use them with validation
            admin_chunk_size = custom_chunk_size if custom_chunk_size is not None else DEFAULT_CHUNK_SIZE
            admin_chunk_overlap = custom_chunk_overlap if custom_chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
            
            # Validate admin settings
            self.chunk_size, self.chunk_overlap = validate_chunk_settings(admin_chunk_size, admin_chunk_overlap)
            
            logger.info(f"🔧 ADMIN SETTINGS: Using custom chunk settings")
            logger.info(f"📏 Requested: size={admin_chunk_size}, overlap={admin_chunk_overlap}")
            logger.info(f"✅ Validated: size={self.chunk_size}, overlap={self.chunk_overlap}")
            
        else:
            # Use default settings from config
            self.chunk_size = DEFAULT_CHUNK_SIZE
            self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
            logger.info(f"📋 DEFAULTS: Using default chunk settings")
        
        logger.info(f"Initializing with faculty={faculty}, collection={self.collection_name}")
        logger.info(f"Chunk settings: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL
        )
        
        # Initialize text splitter with current settings - PURE SIZE-ONLY
        self._create_text_splitter()
        
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _create_text_splitter(self):
        """
        Create text splitter with current chunk settings
        NEW: Uses PURE SIZE-ONLY splitting with no separators at all
        """
        self.text_splitter = PureSizeTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"🔤 Text splitter created: PURE SIZE-ONLY splitting with size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def _get_collection_name(self, faculty):
        """Generate collection name based on faculty"""
        if faculty:
            return f"{COLLECTION_NAME}_{faculty}"
        return COLLECTION_NAME
    
    @staticmethod
    def get_all_faculty_collections():
        """Get list of all faculty collections"""
        collections = []
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            all_collections = client.list_collections()
            
            # Valid faculties from config plus general
            valid_faculties = set(FACULTIES + ["general"])
            
            for collection in all_collections:
                collection_name = collection.name
                
                # Skip collections that don't match our naming pattern
                if not collection_name.startswith(COLLECTION_NAME):
                    continue
                
                if "_" in collection_name and collection_name.startswith(f"{COLLECTION_NAME}_"):
                    # Extract faculty name after the base collection name
                    faculty = collection_name.replace(f"{COLLECTION_NAME}_", "", 1)
                    
                    # Only include valid faculties (avoid duplicates like 'documents_arts')
                    if faculty in valid_faculties and not faculty.startswith("documents"):
                        collections.append({
                            "faculty": faculty,
                            "collection_name": collection_name,
                            "count": collection.count()
                        })
                elif collection_name == COLLECTION_NAME:
                    # Base collection (general)
                    collections.append({
                        "faculty": "general",
                        "collection_name": collection_name,
                        "count": collection.count()
                    })
        except Exception as e:
            logger.error(f"Error getting faculty collections: {e}")
        
        return collections
    
    @staticmethod
    def get_available_faculties():
        """Get list of available faculties from config and existing collections"""
        # Start with configured faculties
        available = set(FACULTIES)
        
        # Add faculties from existing collections
        collections = EmbeddingService.get_all_faculty_collections()
        for collection_info in collections:
            if collection_info["faculty"] != "general":
                available.add(collection_info["faculty"])
        
        return sorted(list(available))
    
    def update_chunk_settings(self, chunk_size=None, chunk_overlap=None):
        """
        ADMIN FUNCTION: Update chunk settings dynamically and reinitialize text splitter
        This allows admin to change chunk settings on-the-fly
        """
        settings_changed = False
        
        if chunk_size is not None:
            old_size = self.chunk_size
            self.chunk_size = chunk_size
            settings_changed = True
            logger.info(f"🔧 ADMIN UPDATE: Chunk size {old_size} → {chunk_size}")
        
        if chunk_overlap is not None:
            old_overlap = self.chunk_overlap
            self.chunk_overlap = chunk_overlap
            settings_changed = True
            logger.info(f"🔧 ADMIN UPDATE: Chunk overlap {old_overlap} → {chunk_overlap}")
        
        if settings_changed:
            # Validate the new settings
            self.chunk_size, self.chunk_overlap = validate_chunk_settings(self.chunk_size, self.chunk_overlap)
            
            # Recreate text splitter with new settings
            self._create_text_splitter()
            
            logger.info(f"✅ ADMIN SETTINGS APPLIED: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "settings_changed": settings_changed
        }
    
    def _initialize_vectorstore(self):
        """Initialize Chroma vector store"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            logger.info(f"Vector store initialized for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _check_file_size(self, file_path: str) -> bool:
        """Check if file size is within limits"""
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                logger.warning(f"File {file_path} is {size_mb:.1f}MB, exceeds limit of {MAX_FILE_SIZE_MB}MB")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking file size for {file_path}: {e}")
            return False
    
    def load_document(self, file_path: str) -> List:
        """Load document based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        # Check file size
        if not self._check_file_size(file_path):
            return []
        
        # Updated loaders
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader
        }
        
        if file_extension not in SUPPORTED_EXTENSIONS:
            if file_extension == '.docx':
                logger.warning(f"DOCX files not supported in current LangChain version. Skipping {file_path}")
                return []
            logger.warning(f"Unsupported file type {file_extension}. Skipping {file_path}")
            return []
        
        try:
            loader = loaders[file_extension](file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document(s) from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def add_single_document(self, file_path: str, filename: str = None):
        """
        Add a single document to the vector store
        USES CURRENT ADMIN CHUNK SETTINGS WITH PURE SIZE-ONLY CHUNKING
        FIXED: Combines all pages into one text before chunking
        """
        try:
            # Load the document
            documents = self.load_document(file_path)
            if not documents:
                logger.warning(f"No content loaded from: {file_path}")
                return False
        
            display_filename = filename or Path(file_path).name
            
            # FIXED: Combine all pages into one large text before chunking
            logger.info(f"📄 DOCUMENT INFO: '{display_filename}'")
            logger.info(f"   - Pages loaded: {len(documents)}")
            
            # Combine all page content into one large text
            combined_text = ""
            page_lengths = []
            
            for i, doc in enumerate(documents):
                page_content = doc.page_content.strip()
                page_lengths.append(len(page_content))
                
                # Add page separator if not empty
                if page_content:
                    if combined_text:
                        combined_text += "\n\n"  # Add separator between pages
                    combined_text += page_content
                
                logger.debug(f"   - Page {i+1}: {len(page_content):,} characters")
            
            total_content_length = len(combined_text)
            logger.info(f"   - Individual page lengths: {page_lengths}")
            logger.info(f"   - Combined content length: {total_content_length:,} characters")
            logger.info(f"   - Expected chunks (~{self.chunk_size:,} chars each): ~{total_content_length // self.chunk_size + 1}")
            
            # Create one document with combined content
            combined_document = Document(
                page_content=combined_text,
                metadata={
                    'filename': display_filename,
                    'filepath': file_path,
                    'faculty': self.faculty or 'general',
                    'pages': len(documents),
                    'total_length': total_content_length
                }
            )
            
            # Split using PURE SIZE-ONLY ADMIN SETTINGS on the COMBINED text
            logger.info(f"🔤 Splitting COMBINED document '{display_filename}' with PURE SIZE-ONLY ADMIN SETTINGS:")
            logger.info(f"   - Chunk size: {self.chunk_size:,} characters")
            logger.info(f"   - Chunk overlap: {self.chunk_overlap:,} characters")
            logger.info(f"   - Input text length: {total_content_length:,} characters")
            
            # Use split_documents on the single combined document
            texts = self.text_splitter.split_documents([combined_document])
            
            # Detailed chunk analysis
            if texts:
                chunk_lengths = [len(chunk.page_content) for chunk in texts]
                total_chunk_content = sum(chunk_lengths)
                avg_length = sum(chunk_lengths) / len(chunk_lengths)
                max_length = max(chunk_lengths)
                min_length = min(chunk_lengths)
                
                # Check for content loss
                content_ratio = total_chunk_content / total_content_length if total_content_length > 0 else 0
                
                logger.info(f"📊 CHUNKING RESULTS for '{display_filename}':")
                logger.info(f"   - Chunks created: {len(texts)}")
                logger.info(f"   - Average chunk length: {avg_length:.0f} chars")
                logger.info(f"   - Min/Max chunk length: {min_length:,}/{max_length:,} chars")
                logger.info(f"   - Total chunk content: {total_chunk_content:,} chars")
                logger.info(f"   - Content preservation: {content_ratio:.1%}")
                logger.info(f"   - Admin target size: {self.chunk_size:,} chars")
                
                # Success indicators
                if avg_length > self.chunk_size * 0.8:
                    logger.info(f"✅ SUCCESS: Chunks are close to target size!")
                elif avg_length > self.chunk_size * 0.5:
                    logger.info(f"⚠️  GOOD: Chunks are reasonably sized")
                else:
                    logger.warning(f"⚠️  Average chunk size ({avg_length:.0f}) is smaller than expected")
                    logger.warning(f"   This might be due to document length or content structure")
                
                # Warning if significant content loss
                if content_ratio < 0.95:
                    logger.warning(f"⚠️  Potential content loss detected: {content_ratio:.1%} of original content preserved")
                
                # Show sample chunks for debugging
                logger.info(f"📝 SAMPLE CHUNKS (first 3):")
                for i, chunk in enumerate(texts[:3]):
                    content_preview = chunk.page_content[:100].replace('\n', ' ').replace('\r', ' ')
                    content_ending = chunk.page_content[-100:].replace('\n', ' ').replace('\r', ' ')
                    logger.info(f"   Chunk {i+1}: {len(chunk.page_content):,} chars")
                    logger.info(f"      Starts: '{content_preview}...'")
                    logger.info(f"      Ends: '...{content_ending}'")
            
            else:
                logger.error(f"❌ No chunks created from document '{display_filename}'")
                return False
            
            # Store in vector database
            logger.info(f"💾 Storing {len(texts)} chunks in vector database...")
            self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            
            logger.info(f"✅ Successfully added document: {display_filename}")
            logger.info(f"   - {len(texts)} chunks stored in {self.faculty or 'general'} faculty")
            logger.info(f"   - Using PURE SIZE-ONLY admin settings: {self.chunk_size:,}/{self.chunk_overlap:,}")
            logger.info(f"   - Original pages: {len(documents)}, Combined into: {len(texts)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding document {file_path}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def remove_document_by_filename(self, filename: str) -> bool:
        """Remove all chunks of a document by filename"""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            collection = client.get_collection(self.collection_name)
            
            # Get all documents with this filename
            results = collection.get(where={"filename": filename})
            
            if results and results['ids']:
                # Delete all chunks with this filename
                collection.delete(where={"filename": filename})
                logger.info(f"Removed {len(results['ids'])} chunks for file: {filename} from {self.faculty or 'general'} faculty")
                
                # Reinitialize vectorstore to reflect changes
                self._initialize_vectorstore()
                return True
            else:
                logger.warning(f"No chunks found for file: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing document {filename}: {e}")
            return False
    
    def list_stored_documents(self) -> List[Dict]:
        """List all unique documents stored in the collection"""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            collection = client.get_collection(self.collection_name)
            
            # Get all metadata
            results = collection.get()
            
            if not results or not results['metadatas']:
                return []
            
            # Extract unique filenames with chunk counts
            filename_counts = {}
            for metadata in results['metadatas']:
                if metadata and 'filename' in metadata:
                    filename = metadata['filename']
                    filename_counts[filename] = filename_counts.get(filename, 0) + 1
            
            # Convert to list of dicts
            documents = []
            for filename, chunk_count in filename_counts.items():
                documents.append({
                    'filename': filename,
                    'chunk_count': chunk_count,
                    'faculty': self.faculty or 'general'
                })
            
            # Sort by filename
            documents.sort(key=lambda x: x['filename'])
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents for {self.faculty or 'general'}: {e}")
            return []
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the collection"""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            client.delete_collection(self.collection_name)
            
            # Reinitialize vectorstore
            self._initialize_vectorstore()
            
            logger.info(f"All documents cleared from collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all documents from {self.collection_name}: {e}")
            return False
    
    def get_document_chunks(self, filename: str, limit: int = 5) -> List[Dict]:
        """Get sample chunks for a specific document"""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            collection = client.get_collection(self.collection_name)
            
            # Get chunks for this filename
            results = collection.get(
                where={"filename": filename},
                limit=limit
            )
            
            chunks = []
            if results and results['documents']:
                for i, (doc_id, content, metadata) in enumerate(zip(
                    results['ids'], results['documents'], results['metadatas']
                )):
                    chunks.append({
                        'id': doc_id,
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'full_content': content,
                        'metadata': metadata,
                        'length': len(content)
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks for {filename}: {e}")
            return []
    
    def search_similar_documents(self, query: str, k: int = None) -> List:
        """Search for similar documents with increased default results"""
        if k is None:
            k = SIMILARITY_SEARCH_K
            
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.debug(f"Search for '{query}' in {self.faculty or 'general'} returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents in {self.faculty or 'general'}: {e}")
            return []
    
    def get_collection_info(self) -> dict:
        """Get information about the collection including current chunk settings"""
        try:
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
            collection = client.get_collection(self.collection_name)
            
            # Count unique documents
            unique_docs = len(self.list_stored_documents())
            
            return {
                "name": collection.name,
                "faculty": self.faculty or 'general',
                "count": collection.count(),
                "unique_documents": unique_docs,
                "metadata": collection.metadata,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "splitting_method": "pure_size_only",
                "admin_configurable": True
            }
        except Exception as e:
            logger.error(f"Error getting collection info for {self.faculty or 'general'}: {e}")
            return {"error": str(e)}
    
    def get_chunk_statistics(self) -> dict:
        """Get statistics about chunks including admin settings info"""
        try:
            info = self.get_collection_info()
            if "error" in info:
                return info
            
            # Get sample chunks to analyze
            sample_results = self.vectorstore.similarity_search("sample", k=min(10, info["count"]))
            
            if sample_results:
                chunk_lengths = [len(result.page_content) for result in sample_results]
                avg_length = sum(chunk_lengths) / len(chunk_lengths)
                min_length = min(chunk_lengths)
                max_length = max(chunk_lengths)
                
                return {
                    "faculty": self.faculty or 'general',
                    "total_chunks": info["count"],
                    "unique_documents": info.get("unique_documents", 0),
                    "configured_chunk_size": self.chunk_size,
                    "configured_overlap": self.chunk_overlap,
                    "average_actual_length": round(avg_length, 1),
                    "min_length": min_length,
                    "max_length": max_length,
                    "sample_size": len(sample_results),
                    "splitting_method": "pure_size_only",
                    "admin_settings_active": True
                }
            else:
                return {"error": "No chunks found for analysis", "faculty": self.faculty or 'general'}
                
        except Exception as e:
            logger.error(f"Error getting chunk statistics for {self.faculty or 'general'}: {e}")
            return {"error": str(e), "faculty": self.faculty or 'general'}

# Usage example
if __name__ == "__main__":
    print("🔧 Faculty-Aware Embedding Service with PURE SIZE-ONLY Admin Dynamic Chunk Settings")
    print("=" * 80)
    
    # Test admin dynamic chunk settings
    print("\n🎛️ Testing PURE SIZE-ONLY Admin Dynamic Chunk Settings:")
    
    # Test with custom admin settings
    admin_chunk_size = 30000
    admin_chunk_overlap = 3000
    
    print(f"Admin requests: chunk_size={admin_chunk_size}, overlap={admin_chunk_overlap}")
    
    # Test faculty collections
    faculties = EmbeddingService.get_available_faculties()
    print(f"\n📚 Available faculties: {faculties}")
    
    collections = EmbeddingService.get_all_faculty_collections()
    print(f"\n📊 Existing collections:")
    for collection_info in collections:
        print(f"  - {collection_info['faculty']}: {collection_info['count']} documents")
    
    # Test with specific faculty and admin settings
    if faculties:
        test_faculty = faculties[0]
        print(f"\n🚀 Testing with faculty: {test_faculty}")
        print(f"🎛️ Using PURE SIZE-ONLY ADMIN SETTINGS: size={admin_chunk_size}, overlap={admin_chunk_overlap}")
        
        embedding_service = EmbeddingService(
            faculty=test_faculty, 
            custom_chunk_size=admin_chunk_size,
            custom_chunk_overlap=admin_chunk_overlap
        )
        
        # Get collection info
        info = embedding_service.get_collection_info()
        print(f"📊 Collection info: {info}")
        
        # Test dynamic settings update
        print(f"\n🔄 Testing dynamic settings update...")
        new_settings = embedding_service.update_chunk_settings(chunk_size=40000, chunk_overlap=4000)
        print(f"Updated settings: {new_settings}")
        
        # Get updated statistics
        stats = embedding_service.get_chunk_statistics()
        print(f"📈 Updated statistics: {stats}")