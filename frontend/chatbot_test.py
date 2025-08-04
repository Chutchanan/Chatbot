import streamlit as st
import os
import sys
from datetime import datetime
import tempfile

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import everything
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from services.chatbot_service import ChatbotService

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = "sqlite:///./chatbot.db"

# Define models directly to avoid import issues
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

# Database setup
def create_tables():
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)

def get_db_session():
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# Initialize database
create_tables()

# Initialize chatbot service
@st.cache_resource
def get_chatbot_service():
    return ChatbotService()

try:
    chatbot_service = get_chatbot_service()
except Exception as e:
    st.error(f"Failed to initialize chatbot: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="University AI Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for chat styling
st.markdown("""
<style>
.user-message {
    background-color: #007bff;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
    margin-left: 20%;
}

.bot-message {
    background-color: #f1f1f1;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: left;
    margin-right: 20%;
}

.file-message {
    background-color: #e3f2fd;
    color: #1976d2;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
    margin-left: 20%;
    border-left: 4px solid #2196f3;
}

.faculty-badge {
    background-color: #4caf50;
    color: white;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    margin-left: 10px;
}

.timestamp {
    font-size: 0.8em;
    color: #666;
    margin-top: 5px;
}

.file-info {
    background-color: #fff3e0;
    padding: 8px;
    border-radius: 6px;
    margin: 5px 0;
    border-left: 3px solid #ff9800;
}
</style>
""", unsafe_allow_html=True)

def authenticate_user(username: str, password: str, db) -> User:
    """Authenticate user with username and password"""
    # Check if user exists
    user = db.query(User).filter(User.username == username).first()
    
    if user:
        # Default password is same as username
        if password == username:
            return user
        else:
            return None
    return None

def main():
    st.title("🎓 University AI Chatbot")
    
    # Sidebar for user authentication
    with st.sidebar:
        st.header("User Login")
        username = st.text_input("Username", placeholder="Enter: demo, user1, or user2")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        if st.button("Login"):
            if username and password:
                db = get_db_session()
                user = authenticate_user(username, password, db)
                if user and user.is_active:
                    # Check if switching users
                    previous_user_id = st.session_state.get('user_id')
                    is_switching_users = previous_user_id is not None and previous_user_id != user.id
                    
                    # Set new user session
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.session_state.faculty = user.faculty or 'general'
                    st.session_state.token_limit = user.token_limit
                    st.session_state.tokens_used = user.tokens_used
                    
                    # Clear previous user's data if switching
                    if is_switching_users:
                        # Clear chat history
                        if 'messages' in st.session_state:
                            del st.session_state['messages']
                        
                        # Clear file memory for previous user
                        try:
                            if previous_user_id:
                                chatbot_service.clear_file_memory(previous_user_id)
                            chatbot_service.clear_memory()
                        except:
                            pass
                        
                        # Clear any file upload states
                        for key in ['show_file_upload', 'new_file_uploaded']:
                            if key in st.session_state:
                                del st.session_state[key]
                    
                    st.success(f"Welcome, {username}!")
                else:
                    st.error("Invalid username or password")
                    st.info("💡 Default password is same as username")
                    st.info("Try: demo/demo, user1/user1, user2/user2")
                db.close()
            else:
                st.error("Please enter both username and password")
        
                # Show user info if logged in
        if 'user_id' in st.session_state:
            st.success(f"Logged in as: {st.session_state.username}")
            st.markdown(f"**Faculty:** `{st.session_state.faculty.title()}`")
            st.info(f"Tokens used: {st.session_state.tokens_used}/{st.session_state.token_limit}")
            
            # Progress bar for token usage (with bounds checking)
            token_percentage = min(100, max(0, (st.session_state.tokens_used / st.session_state.token_limit) * 100))
            st.progress(token_percentage / 100)
            
            if st.button("Logout"):
                # Clear everything when logging out
                for key in ['user_id', 'username', 'faculty', 'token_limit', 'tokens_used']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Clear chat history and file memory
                if 'messages' in st.session_state:
                    del st.session_state['messages']
                
                # Clear chatbot memory for this session
                try:
                    chatbot_service.clear_memory()
                except:
                    pass
                    
                st.rerun()
        
    
    # Main chat interface
    if 'user_id' not in st.session_state:
        st.warning("Please login to start chatting")
        st.info("💡 Login with: demo/demo, user1/user1, or user2/user2")
        
        # Show available faculties info
        st.subheader("🎓 Available Faculties")
        st.write("Each faculty has its own knowledge base:")
        st.write("• **Engineering** - Technical documentation and specs")
        st.write("• **Business** - Policies and procedures")
        st.write("• **Science** - Research papers and data")
        st.write("• **Arts** - Creative and cultural content")
        st.write("• **General** - Common university information")
        
        return
    
    # Faculty header with better contrast
    st.markdown(f"""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #4CAF50;">
        <h3 style="color: #2c3e50; margin: 0 0 10px 0;">🤖 AI Study Assistant - {st.session_state.faculty.title()} Faculty</h3>
        <p style="color: #34495e; margin: 0; font-size: 16px;">Ask me anything! I can help with studies, assignments, general questions, and I'll use university documents when available.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history (user-specific)
    chat_history_key = f'messages_{st.session_state.user_id}'
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []
    
    # Use user-specific chat history
    current_messages = st.session_state[chat_history_key]
    
    # File upload section with memory management
    st.subheader("📎 Upload File (Optional)")
    
    # Check if user has a file in memory
    file_in_memory = None
    if 'user_id' in st.session_state:
        try:
            file_in_memory = chatbot_service.get_uploaded_file_info(st.session_state.user_id)
        except:
            file_in_memory = None
    
    if file_in_memory:
        # Show current file in memory
        st.success(f"📁 **Current file in memory:** {file_in_memory['filename']}")
        st.info(f"📊 Type: {file_in_memory['type']} | Size: {file_in_memory['size']} bytes | Uploaded: {file_in_memory['upload_time']}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Remove Current File"):
                chatbot_service.clear_file_memory(st.session_state.user_id)
                st.success("File removed from memory!")
                st.rerun()
        
        with col2:
            if st.button("⬆️ Upload New File"):
                st.session_state.show_file_upload = True
                st.rerun()
        
        # Only show file uploader if user wants to upload new file
        if st.session_state.get('show_file_upload', False):
            st.warning("⚠️ Uploading a new file will replace the current one in memory.")
            uploaded_file = st.file_uploader(
                "Choose new file",
                type=['pdf', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'webp'],
                help="This will replace your current file"
            )
            
            if uploaded_file:
                st.info(f"**New file selected:** {uploaded_file.name} ({uploaded_file.size} bytes)")
                if st.button("✅ Replace File"):
                    # Clear the flag and let the main chat logic handle the new file
                    st.session_state.show_file_upload = False
                    st.session_state.new_file_uploaded = uploaded_file
                    st.rerun()
            
            if st.button("❌ Cancel"):
                st.session_state.show_file_upload = False
                st.rerun()
        else:
            uploaded_file = None
    else:
        # No file in memory, show regular upload
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a file to ask questions about it",
                type=['pdf', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'webp'],
                help="Supported: PDF, Text files, Images (JPG, PNG, etc.)"
            )
        
        with col2:
            if uploaded_file:
                st.write("**File Info:**")
                st.write(f"Name: {uploaded_file.name}")
                st.write(f"Size: {uploaded_file.size} bytes")
                st.write(f"Type: {uploaded_file.type}")
    
    # Handle new file from replacement
    if st.session_state.get('new_file_uploaded'):
        uploaded_file = st.session_state.new_file_uploaded
        del st.session_state.new_file_uploaded
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in current_messages:
            if message['role'] == 'user':
                if message.get('has_file', False):
                    st.markdown(f"""
                    <div class="file-message">
                        📎 {message['content']}
                        <div class="file-info">
                            <strong>File:</strong> {message.get('filename', 'Unknown')} 
                            ({message.get('file_size', 0)} bytes)
                        </div>
                        <div class="timestamp">{message['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="user-message">
                        {message['content']}
                        <div class="timestamp">{message['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    {message['content']}
                    <span class="faculty-badge">{st.session_state.faculty}</span>
                    <div class="timestamp">{message['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input or uploaded_file:
        if user_input:  # Only process if there's actual user input
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Check if user has file in memory or is uploading new file
            has_memory_file = chatbot_service.has_uploaded_file(st.session_state.user_id) if 'user_id' in st.session_state else False
            
            # Determine if we're processing a file or regular chat
            if uploaded_file or has_memory_file:
                # File-based chat (new upload or using memory)
                if uploaded_file:
                    # New file uploaded
                    current_messages.append({
                        'role': 'user',
                        'content': user_input,
                        'timestamp': timestamp,
                        'has_file': True,
                        'filename': uploaded_file.name,
                        'file_size': uploaded_file.size
                    })
                    
                    with st.spinner("Processing new file and generating response..."):
                        db = get_db_session()
                        file_content = uploaded_file.read()
                        response = chatbot_service.chat_with_file(
                            st.session_state.user_id, 
                            user_input, 
                            uploaded_file,
                            file_content,
                            db
                        )
                        
                else:
                    # Using file from memory
                    file_info = chatbot_service.get_uploaded_file_info(st.session_state.user_id)
                    current_messages.append({
                        'role': 'user',
                        'content': user_input,
                        'timestamp': timestamp,
                        'has_file': True,
                        'filename': file_info['filename'],
                        'file_size': file_info['size']
                    })
                    
                    with st.spinner("Generating response using uploaded file..."):
                        db = get_db_session()
                        response = chatbot_service.chat_with_memory_file(
                            st.session_state.user_id, 
                            user_input,
                            db
                        )
                
                if 'error' in response:
                    bot_response = f"Error: {response['error']}"
                else:
                    bot_response = response['response']
                    # Update token usage
                    st.session_state.tokens_used = st.session_state.token_limit - response['remaining_tokens']
                    
                    # Show file processing info
                    if 'file_processed' in response:
                        file_info = response['file_processed']
                        memory_indicator = " (from memory)" if file_info.get('from_memory') else " (newly processed)"
                        bot_response += f"\n\n📎 **File used:** {file_info['filename']} ({file_info['type']}){memory_indicator}"
                        
            else:
                # Regular chat without file
                current_messages.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': timestamp,
                    'has_file': False
                })
                
                # Get bot response from faculty-specific RAG
                with st.spinner("Thinking..."):
                    db = get_db_session()
                    response = chatbot_service.chat(st.session_state.user_id, user_input, db)
                    
                    if 'error' in response:
                        bot_response = f"Error: {response['error']}"
                    else:
                        bot_response = response['response']
                        # Update token usage
                        st.session_state.tokens_used = st.session_state.token_limit - response['remaining_tokens']
                        
                        # Note: Sources removed for cleaner interface
                        # The AI response already incorporates university knowledge when available
                        
            
            # Add bot response to history
            current_messages.append({
                'role': 'bot',
                'content': bot_response,
                'timestamp': timestamp
            })
            
            # Update session state
            st.session_state[chat_history_key] = current_messages
            
            db.close()
            st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Chat"):
            # Clear user-specific chat history
            if chat_history_key in st.session_state:
                del st.session_state[chat_history_key]
            chatbot_service.clear_memory(st.session_state.user_id)
            st.success("Chat and file memory cleared!")
            st.rerun()
    
    with col2:
        if st.button("Show Token Usage"):
            db = get_db_session()
            try:
                # Get recent chat history
                from services.chatbot_service import ChatLog
                recent_chats = db.query(ChatLog).filter(
                    ChatLog.user_id == st.session_state.user_id
                ).order_by(ChatLog.timestamp.desc()).limit(5).all()
                
                if recent_chats:
                    st.write("**Recent Token Usage:**")
                    for chat in recent_chats:
                        file_icon = "📎" if chat.has_file_upload else "💬"
                        st.write(f"{file_icon} {chat.tokens_used} tokens - {chat.timestamp.strftime('%m/%d %H:%M')}")
                else:
                    st.info("No recent chats found")
            except Exception as e:
                st.error(f"Error loading token usage: {e}")
            finally:
                db.close()
    
    with col3:
        if st.button("Faculty Info"):
            st.info(f"""
            **Faculty:** {st.session_state.faculty.title()}
            
            You can only access documents and knowledge from your assigned faculty. 
            
            **Features:**
            • Ask questions about faculty documents
            • Upload files for analysis
            • Get faculty-specific responses
            """)
    
    # Show example questions if no messages
    if not current_messages:
        st.markdown("### 💡 Ask me anything!")
        
        # General examples for all students
        st.markdown("**📚 Academic Help:**")
        st.markdown("- How do I write a good research paper?")
        st.markdown("- What's the difference between qualitative and quantitative research?")
        st.markdown("- Can you help me solve this math problem?")
        st.markdown("- Explain machine learning in simple terms")
        
        st.markdown("**🎓 University-Specific (when documents are available):**")
        # Faculty-specific examples based on uploaded documents
        if st.session_state.faculty == "engineering":
            st.markdown("- What programming languages are taught here?")
            st.markdown("- Tell me about the robotics lab facilities")
        elif st.session_state.faculty == "business":
            st.markdown("- What MBA programs do we offer?")
            st.markdown("- How does the business incubator work?")
        elif st.session_state.faculty == "science":
            st.markdown("- What research projects are currently running?")
            st.markdown("- Tell me about the observatory facilities")
        elif st.session_state.faculty == "arts":
            st.markdown("- What creative programs are available?")
            st.markdown("- When is the next arts festival?")
        else:
            st.markdown("- What are the university's admission requirements?")
            st.markdown("- How do I contact student services?")
        
        st.markdown("**🤔 General Questions:**")
        st.markdown("- How can I improve my study habits?")
        st.markdown("- What career options are there in my field?")
        st.markdown("- Help me plan my course schedule")
        st.markdown("- Tips for managing stress during exams")
    
    # Footer with updated description
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        🤖 AI Study Assistant for <strong>{st.session_state.faculty.title()} Faculty</strong>
        | 📚 Enhanced with University Knowledge Base
        | 🔍 Powered by GPT-4o-mini with Vision
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()