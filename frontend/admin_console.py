import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import everything we need
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from services.embedding_service import EmbeddingService

try:
    from config import ADMIN_USERNAME, ADMIN_PASSWORD, DEFAULT_TOKEN_LIMIT, DATABASE_URL
except ImportError:
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"
    DEFAULT_TOKEN_LIMIT = 1000
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

class ChatLog(Base):
    __tablename__ = "chat_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    query = Column(Text)
    response = Column(Text)
    tokens_used = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Database setup
def get_db():
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def authenticate_admin(username: str, password: str) -> bool:
    """Authenticate admin user"""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def create_user(username: str, email: str, token_limit: int) -> bool:
    """Create new user"""
    db = get_db()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return False
        
        new_user = User(
            username=username,
            email=email,
            token_limit=token_limit,
            tokens_used=0,
            is_active=True
        )
        
        db.add(new_user)
        db.commit()
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False
    finally:
        db.close()

# Page config
st.set_page_config(
    page_title="Admin Console",
    page_icon="⚙️",
    layout="wide"
)

def main():
    st.title("⚙️ Admin Console")
    
    # Admin authentication
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.subheader("Admin Login")
        col1, col2 = st.columns(2)
        
        with col1:
            admin_username = st.text_input("Username")
            admin_password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if authenticate_admin(admin_username, admin_password):
                    st.session_state.admin_authenticated = True
                    st.success("Admin authenticated successfully!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
                    st.info(f"Try: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "User Management", "Chat Logs", "Document Management"]
    )
    
    if st.sidebar.button("Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "User Management":
        show_user_management()
    elif page == "Chat Logs":
        show_chat_logs()
    elif page == "Document Management":
        show_document_management()

def show_dashboard():
    """Show dashboard with statistics"""
    st.header("📊 Dashboard")
    
    db = get_db()
    
    try:
        # Get statistics
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        total_chats = db.query(ChatLog).count()
        total_tokens_used = db.query(func.sum(ChatLog.tokens_used)).scalar() or 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", total_users)
        
        with col2:
            st.metric("Active Users", active_users)
        
        with col3:
            st.metric("Total Chats", total_chats)
        
        with col4:
            st.metric("Total Tokens Used", total_tokens_used)
        
        # Charts
        st.subheader("📈 Analytics")
        
        # User token usage chart
        users = db.query(User).all()
        if users:
            user_data = pd.DataFrame([{
                'username': user.username,
                'tokens_used': user.tokens_used,
                'token_limit': user.token_limit,
                'usage_percent': (user.tokens_used / user.token_limit * 100) if user.token_limit > 0 else 0
            } for user in users])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(user_data, x='username', y=['tokens_used', 'token_limit'],
                           title='Token Usage by User')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(user_data, values='tokens_used', names='username',
                           title='Token Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Chat activity over time
        chat_logs = db.query(ChatLog).all()
        if chat_logs:
            chat_data = pd.DataFrame([{
                'date': log.timestamp.date(),
                'hour': log.timestamp.hour,
                'tokens': log.tokens_used
            } for log in chat_logs])
            
            daily_activity = chat_data.groupby('date').size().reset_index(name='chat_count')
            
            fig = px.line(daily_activity, x='date', y='chat_count',
                         title='Daily Chat Activity')
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
    finally:
        db.close()

def show_user_management():
    """Show user management interface"""
    st.header("👥 User Management")
    
    tab1, tab2, tab3 = st.tabs(["All Users", "Create User", "Edit User"])
    
    with tab1:
        st.subheader("All Users")
        db = get_db()
        
        try:
            users = db.query(User).all()
            
            if users:
                user_data = pd.DataFrame([{
                    'ID': user.id,
                    'Username': user.username,
                    'Email': user.email,
                    'Token Limit': user.token_limit,
                    'Tokens Used': user.tokens_used,
                    'Usage %': f"{(user.tokens_used/user.token_limit*100):.1f}%" if user.token_limit > 0 else "0%",
                    'Active': "✅" if user.is_active else "❌",
                    'Created': user.created_at.strftime("%Y-%m-%d")
                } for user in users])
                
                st.dataframe(user_data, use_container_width=True)
            else:
                st.info("No users found")
        except Exception as e:
            st.error(f"Error loading users: {e}")
        finally:
            db.close()
    
    with tab2:
        st.subheader("Create New User")
        
        with st.form("create_user_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            token_limit = st.number_input("Token Limit", value=DEFAULT_TOKEN_LIMIT, min_value=100)
            
            if st.form_submit_button("Create User"):
                if username and email:
                    if create_user(username, email, token_limit):
                        st.success(f"User '{username}' created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create user. Username or email might already exist.")
                else:
                    st.error("Please fill in all fields")
    
    with tab3:
        st.subheader("Edit User")
        
        db = get_db()
        try:
            users = db.query(User).all()
            if users:
                user_options = {f"{user.username} ({user.email})": user.id for user in users}
                selected_user = st.selectbox("Select User", list(user_options.keys()))
                
                if selected_user:
                    user_id = user_options[selected_user]
                    user = db.query(User).filter(User.id == user_id).first()
                    
                    with st.form("edit_user_form"):
                        new_token_limit = st.number_input("Token Limit", value=user.token_limit)
                        reset_tokens = st.checkbox("Reset used tokens to 0")
                        is_active = st.checkbox("Active", value=user.is_active)
                        
                        if st.form_submit_button("Update User"):
                            user.token_limit = new_token_limit
                            user.is_active = is_active
                            
                            if reset_tokens:
                                user.tokens_used = 0
                            
                            db.commit()
                            st.success("User updated successfully!")
                            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            db.close()

def show_chat_logs():
    """Show chat logs interface"""
    st.header("💬 Chat Logs")
    
    db = get_db()
    
    try:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            users = db.query(User).all()
            user_options = ["All Users"] + [user.username for user in users]
            selected_user = st.selectbox("Filter by User", user_options)
        
        with col2:
            days_back = st.number_input("Days back", value=7, min_value=1)
        
        with col3:
            limit = st.number_input("Max records", value=50, min_value=10)
        
        # Query chat logs
        query = db.query(ChatLog, User).join(User, ChatLog.user_id == User.id)
        
        if selected_user != "All Users":
            query = query.filter(User.username == selected_user)
        
        # Date filter
        date_filter = datetime.now() - timedelta(days=days_back)
        query = query.filter(ChatLog.timestamp >= date_filter)
        
        chat_logs = query.order_by(ChatLog.timestamp.desc()).limit(limit).all()
        
        if chat_logs:
            st.subheader(f"Found {len(chat_logs)} chat logs")
            
            for chat_log, user in chat_logs:
                with st.expander(f"{user.username} - {chat_log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.write("**Query:**", chat_log.query)
                    st.write("**Response:**", chat_log.response)
                    st.write("**Tokens Used:**", chat_log.tokens_used)
        else:
            st.info("No chat logs found for the selected criteria")
    
    except Exception as e:
        st.error(f"Error loading chat logs: {e}")
    finally:
        db.close()

def show_document_management():
    """Show document management interface"""
    st.header("📄 Document Management")
    
    # Initialize embedding service
    try:
        embedding_service = EmbeddingService()
        
        # Collection info
        st.subheader("Collection Information")
        collection_info = embedding_service.get_collection_info()
        
        if "error" not in collection_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Collection Name", collection_info.get("name", "N/A"))
            with col2:
                st.metric("Document Count", collection_info.get("count", 0))
            with col3:
                st.metric("Chunk Size", collection_info.get("chunk_size", "N/A"))
        else:
            st.error(f"Error getting collection info: {collection_info['error']}")
        
        # Upload new documents
        st.subheader("Upload New Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'csv']
        )
        
        if uploaded_files:
            if st.button("Process and Upload Documents"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Add to vector store
                        embedding_service.add_single_document(temp_path)
                        st.success(f"Successfully processed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Upload completed!")
                st.rerun()
        
        # Test search functionality
        st.subheader("Test Search")
        
        test_query = st.text_input("Enter test query")
        
        if test_query:
            if st.button("Search"):
                try:
                    results = embedding_service.search_similar_documents(test_query, k=3)
                    
                    if results:
                        st.write("**Search Results:**")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}"):
                                st.write(result.page_content)
                    else:
                        st.info("No results found")
                except Exception as e:
                    st.error(f"Search error: {e}")
        
    except Exception as e:
        st.error(f"Error initializing document management: {e}")

if __name__ == "__main__":
    main()