import os
import logging
from typing import List, Optional, Dict
from pathlib import Path
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

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
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MAX_CHUNKS_PER_DOCUMENT,
        SIMILARITY_SEARCH_K,
        EMBEDDING_MODEL,
        BATCH_SIZE,
        MAX_RETRIES,
        LOG_LEVEL,
        MAX_FILE_SIZE_MB,
        SUPPORTED_EXTENSIONS,
        FACULTIES
    )
except ImportError:
    # Fallback config if import fails
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "company_documents"
    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 500
    MAX_CHUNKS_PER_DOCUMENT = 1000
    SIMILARITY_SEARCH_K = 3
    EMBEDDING_MODEL = "text-embedding-ada-002"
    BATCH_SIZE = 10
    MAX_RETRIES = 3
    LOG_LEVEL = "INFO"
    MAX_FILE_SIZE_MB = 10
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.csv']
    FACULTIES = ["engineering", "business", "science", "arts"]

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, faculty=None, custom_chunk_size=None, custom_chunk_overlap=None):
        """
        Initialize embedding service with faculty support
        
        Args:
            faculty: Faculty name for faculty-specific collections
            custom_chunk_size: Override chunk size from config
            custom_chunk_overlap: Override chunk overlap from config
        """
        # Set faculty and collection name
        self.faculty = faculty
        self.collection_name = self._get_collection_name(faculty)
        
        # Set chunk configuration
        self.chunk_size = custom_chunk_size or CHUNK_SIZE
        self.chunk_overlap = custom_chunk_overlap or CHUNK_OVERLAP
        
        logger.info(f"Initializing with faculty={faculty}, collection={self.collection_name}")
        logger.info(f"Chunk settings: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vectorstore = None
        self._initialize_vectorstore()
    
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
        """Update chunk settings and reinitialize text splitter"""
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        
        # Reinitialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Updated chunk settings: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
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
    
    def process_and_store_documents(self, data_directory: str, reprocess=False):
        """
        Process all documents in directory and store in vector database
        
        Args:
            data_directory: Path to documents directory
            reprocess: If True, clear existing collection before processing
        """
        data_path = Path(data_directory)
        
        if not data_path.exists():
            logger.error(f"Data directory {data_directory} does not exist")
            return
        
        # Clear existing collection if reprocessing
        if reprocess:
            logger.info(f"Clearing existing collection {self.collection_name} for reprocessing...")
            try:
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
                client.delete_collection(self.collection_name)
                # Reinitialize vectorstore
                self._initialize_vectorstore()
            except Exception as e:
                logger.warning(f"Could not clear existing collection: {e}")
        
        all_documents = []
        processed_files = []
        
        # Process all supported files
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    logger.info(f"Processing file: {file_path}")
                    documents = self.load_document(str(file_path))
                    if documents:
                        # Add filename and faculty metadata to each document
                        for doc in documents:
                            if not hasattr(doc, 'metadata'):
                                doc.metadata = {}
                            doc.metadata['filename'] = file_path.name
                            doc.metadata['filepath'] = str(file_path)
                            doc.metadata['faculty'] = self.faculty or 'general'
                        
                        all_documents.extend(documents)
                        processed_files.append(file_path.name)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        if not all_documents:
            logger.warning("No documents found to process")
            return
        
        # Split documents into chunks - respect admin settings
        logger.info(f"Splitting {len(all_documents)} documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        texts = self.text_splitter.split_documents(all_documents)
        
        # Log chunk sizes for debugging
        if texts:
            chunk_lengths = [len(chunk.page_content) for chunk in texts[:5]]  # First 5 chunks
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            logger.info(f"Sample chunk lengths: {chunk_lengths}")
            logger.info(f"Average chunk length: {avg_length:.0f} characters")
        
        # Store all chunks without limits
        logger.info(f"Storing all {len(texts)} chunks in vector database")
        
        # Store in vector database in batches
        batch_size = BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                self.vectorstore.add_documents(batch)
                logger.info(f"Stored batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error storing batch {i//batch_size + 1}: {e}")
        
        # Persist the vectorstore
        self.vectorstore.persist()
        
        logger.info(f"Successfully processed and stored {len(all_documents)} documents from {len(processed_files)} files")
        logger.info(f"Files processed: {', '.join(processed_files)}")
        logger.info(f"Total chunks created: {len(texts)}")
        logger.info(f"Faculty: {self.faculty or 'general'}")
    
    def add_single_document(self, file_path: str, filename: str = None):
        """Add a single document to the vector store"""
        try:
            documents = self.load_document(file_path)
            if documents:
                # Add filename and faculty metadata
                display_filename = filename or Path(file_path).name
                for doc in documents:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['filename'] = display_filename
                    doc.metadata['filepath'] = file_path
                    doc.metadata['faculty'] = self.faculty or 'general'
                
                texts = self.text_splitter.split_documents(documents)
                self.vectorstore.add_documents(texts)
                self.vectorstore.persist()
                logger.info(f"Successfully added document: {display_filename} ({len(texts)} chunks) to {self.faculty or 'general'} faculty")
                return True
            else:
                logger.warning(f"No content loaded from: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
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
                        'metadata': metadata
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks for {filename}: {e}")
            return []
    
    def search_similar_documents(self, query: str, k: int = None) -> List:
        """Search for similar documents with increased default results"""
        if k is None:
            k = SIMILARITY_SEARCH_K  # Now defaults to 8 instead of 3
            
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.debug(f"Search for '{query}' in {self.faculty or 'general'} returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents in {self.faculty or 'general'}: {e}")
            return []
    
    def get_collection_info(self) -> dict:
        """Get information about the collection"""
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
                "chunk_overlap": self.chunk_overlap
            }
        except Exception as e:
            logger.error(f"Error getting collection info for {self.faculty or 'general'}: {e}")
            return {"error": str(e)}
    
    def get_chunk_statistics(self) -> dict:
        """Get statistics about chunks"""
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
                    "sample_size": len(sample_results)
                }
            else:
                return {"error": "No chunks found for analysis", "faculty": self.faculty or 'general'}
                
        except Exception as e:
            logger.error(f"Error getting chunk statistics for {self.faculty or 'general'}: {e}")
            return {"error": str(e), "faculty": self.faculty or 'general'}

# Usage example
if __name__ == "__main__":
    print("🔧 Faculty-Aware Embedding Service Test")
    print("=" * 40)
    
    # Test faculty collections
    faculties = EmbeddingService.get_available_faculties()
    print(f"\n📚 Available faculties: {faculties}")
    
    collections = EmbeddingService.get_all_faculty_collections()
    print(f"\n📊 Existing collections:")
    for collection_info in collections:
        print(f"  - {collection_info['faculty']}: {collection_info['count']} documents")
    
    # Test with specific faculty
    if faculties:
        test_faculty = faculties[0]
        print(f"\n🚀 Testing with faculty: {test_faculty}")
        embedding_service = EmbeddingService(faculty=test_faculty)
        
        # Get collection info
        info = embedding_service.get_collection_info()
        print(f"📊 Collection info: {info}")