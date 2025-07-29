import streamlit as st
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import everything
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from services.chatbot_service import ChatbotService
import hashlib

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
    page_title="AI Chatbot",
    page_icon="🤖",
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

.timestamp {
    font-size: 0.8em;
    color: #666;
    margin-top: 5px;
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
    st.title("🤖 AI Chatbot")
    
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
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.session_state.token_limit = user.token_limit
                    st.session_state.tokens_used = user.tokens_used
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
            st.info(f"Tokens used: {st.session_state.tokens_used}/{st.session_state.token_limit}")
            
            if st.button("Logout"):
                for key in ['user_id', 'username', 'token_limit', 'tokens_used']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Main chat interface
    if 'user_id' not in st.session_state:
        st.warning("Please login to start chatting")
        st.info("💡 Login with: demo/demo, user1/user1, or user2/user2")
        return
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
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
                    <div class="timestamp">{message['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': timestamp
        })
        
        # Get bot response
        with st.spinner("Thinking..."):
            db = get_db_session()
            response = chatbot_service.chat(st.session_state.user_id, user_input, db)
            
            if 'error' in response:
                bot_response = f"Error: {response['error']}"
            else:
                bot_response = response['response']
                # Update token usage
                st.session_state.tokens_used = st.session_state.token_limit - response['remaining_tokens']
            
            # Add bot response to history
            st.session_state.messages.append({
                'role': 'bot',
                'content': bot_response,
                'timestamp': timestamp
            })
            
            db.close()
        
        st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        chatbot_service.clear_memory()
        st.rerun()
    
    # Show some example questions
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        st.markdown("- What is our company about?")
        st.markdown("- What are your main services?")
        st.markdown("- What is the company policy?")
        st.markdown("- How can I contact support?")

if __name__ == "__main__":
    main()