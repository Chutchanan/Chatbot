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
    from config import ADMIN_USERNAME, ADMIN_PASSWORD, DEFAULT_TOKEN_LIMIT, DATABASE_URL, FACULTIES
except ImportError:
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"
    DEFAULT_TOKEN_LIMIT = 1000
    DATABASE_URL = "sqlite:///./chatbot.db"
    FACULTIES = ["engineering", "business", "science", "arts"]

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

# Database setup
def get_db():
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def authenticate_admin(username: str, password: str) -> bool:
    """Authenticate admin user"""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def create_user(username: str, email: str, faculty: str, token_limit: int) -> bool:
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
            faculty=faculty,
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
    page_title="Faculty Admin Console",
    page_icon="🎓",
    layout="wide"
)

def main():
    st.title("🎓 Faculty Admin Console")
    
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
        ["Dashboard", "User Management", "Chat Logs", "Document Management", "Faculty Analytics"]
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
    elif page == "Faculty Analytics":
        show_faculty_analytics()

def show_dashboard():
    """Show dashboard with faculty statistics"""
    st.header("📊 Dashboard")
    
    db = get_db()
    
    try:
        # Get statistics
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        total_chats = db.query(ChatLog).count()
        total_tokens_used = db.query(func.sum(ChatLog.tokens_used)).scalar() or 0
        file_upload_chats = db.query(ChatLog).filter(ChatLog.has_file_upload == True).count()
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Users", total_users)
        
        with col2:
            st.metric("Active Users", active_users)
        
        with col3:
            st.metric("Total Chats", total_chats)
        
        with col4:
            st.metric("Total Tokens Used", total_tokens_used)
        
        with col5:
            st.metric("File Upload Chats", file_upload_chats)
        
        # Faculty breakdown
        st.subheader("👥 Users by Faculty")
        faculty_users = db.query(User.faculty, func.count(User.id).label('user_count')).group_by(User.faculty).all()
        
        if faculty_users:
            faculty_data = pd.DataFrame(faculty_users, columns=['Faculty', 'Users'])
            faculty_data['Faculty'] = faculty_data['Faculty'].fillna('general')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(faculty_data, x='Faculty', y='Users', 
                           title='Users per Faculty')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(faculty_data, values='Users', names='Faculty',
                           title='Faculty Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Token usage by faculty
        st.subheader("💰 Token Usage by Faculty")
        faculty_tokens = db.query(
            User.faculty, 
            func.sum(User.tokens_used).label('total_tokens')
        ).group_by(User.faculty).all()
        
        if faculty_tokens:
            token_data = pd.DataFrame(faculty_tokens, columns=['Faculty', 'Tokens'])
            token_data['Faculty'] = token_data['Faculty'].fillna('general')
            token_data['Tokens'] = token_data['Tokens'].fillna(0)
            
            fig = px.bar(token_data, x='Faculty', y='Tokens',
                       title='Token Usage by Faculty')
            st.plotly_chart(fig, use_container_width=True)
        
        # Document collections
        st.subheader("📚 Document Collections")
        collections = EmbeddingService.get_all_faculty_collections()
        
        if collections:
            collection_data = pd.DataFrame(collections)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(collection_data, x='faculty', y='count',
                           title='Documents per Faculty')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show collection details
                st.write("**Collection Details:**")
                for collection in collections:
                    st.write(f"• **{collection['faculty']}**: {collection['count']} documents")
        else:
            st.info("No document collections found")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
    finally:
        db.close()

def show_user_management():
    """Show user management interface with faculty support"""
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
                    'Faculty': user.faculty or 'general',
                    'Token Limit': user.token_limit,
                    'Tokens Used': user.tokens_used,
                    'Usage %': f"{(user.tokens_used/user.token_limit*100):.1f}%" if user.token_limit > 0 else "0%",
                    'Active': "✅" if user.is_active else "❌",
                    'Created': user.created_at.strftime("%Y-%m-%d")
                } for user in users])
                
                st.dataframe(user_data, use_container_width=True)
                
                # Filter by faculty
                st.subheader("Filter by Faculty")
                available_faculties = ['All'] + list(user_data['Faculty'].unique())
                selected_faculty = st.selectbox("Select Faculty", available_faculties)
                
                if selected_faculty != 'All':
                    filtered_data = user_data[user_data['Faculty'] == selected_faculty]
                    st.dataframe(filtered_data, use_container_width=True)
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
            
            # Faculty selection
            available_faculties = EmbeddingService.get_available_faculties()
            if 'general' not in available_faculties:
                available_faculties.insert(0, 'general')
            
            faculty = st.selectbox("Faculty", available_faculties)
            token_limit = st.number_input("Token Limit", value=DEFAULT_TOKEN_LIMIT, min_value=100)
            
            if st.form_submit_button("Create User"):
                if username and email:
                    if create_user(username, email, faculty, token_limit):
                        st.success(f"User '{username}' created successfully in {faculty} faculty!")
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
                user_options = {f"{user.username} ({user.email}) - {user.faculty or 'general'}": user.id for user in users}
                selected_user = st.selectbox("Select User", list(user_options.keys()))
                
                if selected_user:
                    user_id = user_options[selected_user]
                    user = db.query(User).filter(User.id == user_id).first()
                    
                    with st.form("edit_user_form"):
                        # Faculty selection for editing
                        available_faculties = EmbeddingService.get_available_faculties()
                        if 'general' not in available_faculties:
                            available_faculties.insert(0, 'general')
                        
                        current_faculty_index = available_faculties.index(user.faculty or 'general') if (user.faculty or 'general') in available_faculties else 0
                        new_faculty = st.selectbox("Faculty", available_faculties, index=current_faculty_index)
                        
                        new_token_limit = st.number_input("Token Limit", value=user.token_limit)
                        reset_tokens = st.checkbox("Reset used tokens to 0")
                        is_active = st.checkbox("Active", value=user.is_active)
                        
                        if st.form_submit_button("Update User"):
                            user.faculty = new_faculty
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
    """Show chat logs interface with faculty and file upload filtering"""
    st.header("💬 Chat Logs")
    
    db = get_db()
    
    try:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            users = db.query(User).all()
            user_options = ["All Users"] + [f"{user.username} ({user.faculty or 'general'})" for user in users]
            selected_user = st.selectbox("Filter by User", user_options)
        
        with col2:
            faculty_options = ["All Faculties"] + EmbeddingService.get_available_faculties()
            selected_faculty = st.selectbox("Filter by Faculty", faculty_options)
        
        with col3:
            file_filter = st.selectbox("File Uploads", ["All Chats", "With Files", "Without Files"])
        
        with col4:
            days_back = st.number_input("Days back", value=7, min_value=1)
        
        limit = st.number_input("Max records", value=50, min_value=10)
        
        # Query chat logs with joins
        query = db.query(ChatLog, User).join(User, ChatLog.user_id == User.id)
        
        # Apply filters
        if selected_user != "All Users":
            username = selected_user.split(" (")[0]
            query = query.filter(User.username == username)
        
        if selected_faculty != "All Faculties":
            query = query.filter(User.faculty == selected_faculty)
        
        if file_filter == "With Files":
            query = query.filter(ChatLog.has_file_upload == True)
        elif file_filter == "Without Files":
            query = query.filter(ChatLog.has_file_upload == False)
        
        # Date filter
        date_filter = datetime.now() - timedelta(days=days_back)
        query = query.filter(ChatLog.timestamp >= date_filter)
        
        chat_logs = query.order_by(ChatLog.timestamp.desc()).limit(limit).all()
        
        if chat_logs:
            st.subheader(f"Found {len(chat_logs)} chat logs")
            
            for chat_log, user in chat_logs:
                file_icon = "📎" if chat_log.has_file_upload else "💬"
                faculty_tag = f"[{user.faculty or 'general'}]"
                
                with st.expander(f"{file_icon} {faculty_tag} {user.username} - {chat_log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.write("**Query:**", chat_log.query)
                    st.write("**Response:**", chat_log.response)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Tokens Used:**", chat_log.tokens_used)
                    with col2:
                        st.write("**Faculty:**", user.faculty or 'general')
                    with col3:
                        if chat_log.has_file_upload:
                            st.write("**File Type:**", chat_log.file_type or 'unknown')
        else:
            st.info("No chat logs found for the selected criteria")
    
    except Exception as e:
        st.error(f"Error loading chat logs: {e}")
    finally:
        db.close()

def show_document_management():
    """Enhanced document management interface with faculty selection"""
    st.header("📄 Faculty Document Management")
    
    # Faculty selection at the top
    st.subheader("🎓 Select Faculty")
    available_faculties = EmbeddingService.get_available_faculties()
    if 'general' not in available_faculties:
        available_faculties.insert(0, 'general')
    
    selected_faculty = st.selectbox(
        "Choose faculty to manage documents:",
        available_faculties,
        help="Select which faculty's document collection to manage"
    )
    
    # Initialize embedding service for selected faculty
    try:
        embedding_service = EmbeddingService(faculty=selected_faculty if selected_faculty != 'general' else None)
        
        # Collection info with faculty context
        st.subheader(f"📊 {selected_faculty.title()} Faculty Collection")
        collection_info = embedding_service.get_collection_info()
        
        if "error" not in collection_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Faculty", selected_faculty.title())
            with col2:
                st.metric("Total Chunks", collection_info.get("count", 0))
            with col3:
                st.metric("Unique Documents", collection_info.get("unique_documents", 0))
            with col4:
                st.metric("Chunk Size", collection_info.get("chunk_size", "N/A"))
        else:
            st.error(f"Error getting collection info: {collection_info['error']}")
        
        # Create tabs for different operations
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Stored Documents", "⬆️ Upload Documents", "🔍 Test Search", "⚙️ Settings"])
        
        # Tab 1: List and manage stored documents
        with tab1:
            st.subheader(f"Documents in {selected_faculty.title()} Faculty")
            
            # Refresh button
            if st.button("🔄 Refresh List"):
                st.rerun()
            
            # Get list of stored documents
            stored_docs = embedding_service.list_stored_documents()
            
            if stored_docs:
                st.write(f"**Found {len(stored_docs)} documents in {selected_faculty} faculty:**")
                
                # Display documents with action buttons
                for idx, doc in enumerate(stored_docs):
                    with st.expander(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Filename:** {doc['filename']}")
                            st.write(f"**Chunks:** {doc['chunk_count']}")
                            st.write(f"**Faculty:** {doc['faculty']}")
                        
                        with col2:
                            # Preview chunks button
                            if st.button("👁️ Preview", key=f"preview_{selected_faculty}_{idx}"):
                                st.session_state[f"show_preview_{selected_faculty}_{idx}"] = True
                        
                        with col3:
                            # Delete button with confirmation
                            if st.button("🗑️ Delete", key=f"delete_{selected_faculty}_{idx}", type="secondary"):
                                st.session_state[f"confirm_delete_{selected_faculty}_{idx}"] = True
                        
                        # Show preview if requested
                        if st.session_state.get(f"show_preview_{selected_faculty}_{idx}", False):
                            chunks = embedding_service.get_document_chunks(doc['filename'], limit=3)
                            if chunks:
                                st.write("**Sample chunks:**")
                                for i, chunk in enumerate(chunks[:3], 1):
                                    st.write(f"**Chunk {i}:** {chunk['content']}")
                            else:
                                st.info("No chunks to preview")
                        
                        # Show delete confirmation
                        if st.session_state.get(f"confirm_delete_{selected_faculty}_{idx}", False):
                            st.warning(f"⚠️ Are you sure you want to delete '{doc['filename']}' from {selected_faculty} faculty?")
                            
                            col_confirm, col_cancel = st.columns(2)
                            with col_confirm:
                                if st.button("✅ Yes, Delete", key=f"confirm_{selected_faculty}_{idx}", type="primary"):
                                    if embedding_service.remove_document_by_filename(doc['filename']):
                                        st.success(f"Successfully deleted '{doc['filename']}'")
                                        st.session_state[f"confirm_delete_{selected_faculty}_{idx}"] = False
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete '{doc['filename']}'")
                            
                            with col_cancel:
                                if st.button("❌ Cancel", key=f"cancel_{selected_faculty}_{idx}"):
                                    st.session_state[f"confirm_delete_{selected_faculty}_{idx}"] = False
                                    st.rerun()
            else:
                st.info(f"No documents stored in {selected_faculty} faculty yet.")
                st.write("Upload some documents in the 'Upload Documents' tab to get started!")
        
        # Tab 2: Upload new documents with faculty context
        with tab2:
            st.subheader(f"Upload Documents to {selected_faculty.title()} Faculty")
            
            # Show current faculty context
            st.info(f"📚 Documents will be uploaded to: **{selected_faculty.title()} Faculty**")
            
            # Chunk settings configuration
            st.subheader("🔧 Chunk Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                chunk_size = st.number_input(
                    "Chunk Size", 
                    min_value=100, 
                    max_value=30000,  # Increased from 8000 to 30000
                    value=collection_info.get("chunk_size", 5000),
                    step=500,  # Increased step for easier navigation
                    help="Size of each text chunk in characters (max 30,000)"
                )
            
            with col2:
                chunk_overlap = st.number_input(
                    "Chunk Overlap", 
                    min_value=0, 
                    max_value=min(chunk_size//2, 10000),  # Increased max overlap
                    value=min(collection_info.get("chunk_overlap", 500), chunk_size//2),
                    step=100,  # Increased step
                    help="Overlap between consecutive chunks (max 10,000 or half of chunk size)"
                )
            
            # File upload
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'csv'],
                help="Supported formats: PDF, TXT, CSV"
            )
            
            if uploaded_files:
                st.write(f"**Selected {len(uploaded_files)} file(s) for {selected_faculty} faculty:**")
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size} bytes)")
                
                if st.button("🚀 Process and Upload Documents", type="primary"):
                    # Create embedding service with custom settings for selected faculty
                    custom_embedding_service = EmbeddingService(
                        faculty=selected_faculty if selected_faculty != 'general' else None,
                        custom_chunk_size=chunk_size,
                        custom_chunk_overlap=chunk_overlap
                    )
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    success_count = 0
                    error_count = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                        
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            # Add to vector store with custom settings
                            if custom_embedding_service.add_single_document(temp_path, uploaded_file.name):
                                st.success(f"✅ Successfully processed: {uploaded_file.name}")
                                success_count += 1
                            else:
                                st.warning(f"⚠️ No content processed from: {uploaded_file.name}")
                                error_count += 1
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                            error_count += 1
                        finally:
                            # Clean up temporary file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text(f"Upload completed! ✅ {success_count} successful, ❌ {error_count} errors")
                    
                    if success_count > 0:
                        st.balloons()
                        # Wait a moment then refresh
                        import time
                        time.sleep(1)
                        st.rerun()
        
        # Tab 3: Test search functionality
        with tab3:
            st.subheader(f"🔍 Test Search in {selected_faculty.title()} Faculty")
            
            test_query = st.text_input("Enter test query", placeholder=f"Search in {selected_faculty} faculty documents...")
            
            col1, col2 = st.columns(2)
            with col1:
                search_k = st.number_input("Number of results", min_value=1, max_value=10, value=3)
            
            if test_query:
                if st.button("🔍 Search", type="primary"):
                    with st.spinner(f"Searching {selected_faculty} faculty documents..."):
                        try:
                            results = embedding_service.search_similar_documents(test_query, k=search_k)
                            
                            if results:
                                st.write(f"**Found {len(results)} results in {selected_faculty} faculty:**")
                                for i, result in enumerate(results, 1):
                                    with st.expander(f"Result {i} - Score: {getattr(result, 'score', 'N/A')}"):
                                        st.write("**Content:**")
                                        st.write(result.page_content)
                                        
                                        if hasattr(result, 'metadata') and result.metadata:
                                            st.write("**Source:**")
                                            filename = result.metadata.get('filename', 'Unknown')
                                            faculty = result.metadata.get('faculty', 'Unknown')
                                            st.write(f"📄 {filename} ({faculty} faculty)")
                            else:
                                st.info(f"No results found in {selected_faculty} faculty for your query.")
                                st.write("Try different keywords or check if documents are uploaded correctly.")
                        except Exception as e:
                            st.error(f"Search error: {e}")
        
        # Tab 4: Settings and statistics
        with tab4:
            st.subheader(f"⚙️ {selected_faculty.title()} Faculty Settings")
            
            # Current settings
            stats = embedding_service.get_chunk_statistics()
            
            if "error" not in stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Current Configuration:**")
                    st.write(f"- Faculty: {stats.get('faculty', 'Unknown')}")
                    st.write(f"- Chunk Size: {stats.get('configured_chunk_size', 'N/A')}")
                    st.write(f"- Chunk Overlap: {stats.get('configured_overlap', 'N/A')}")
                    st.write(f"- Total Chunks: {stats.get('total_chunks', 0)}")
                    st.write(f"- Unique Documents: {stats.get('unique_documents', 0)}")
                
                with col2:
                    st.write("**Chunk Statistics:**")
                    st.write(f"- Average Length: {stats.get('average_actual_length', 0):.1f} chars")
                    st.write(f"- Min Length: {stats.get('min_length', 0)} chars")
                    st.write(f"- Max Length: {stats.get('max_length', 0)} chars")
                    st.write(f"- Sample Size: {stats.get('sample_size', 0)} chunks")
                
                # Clear faculty collection
                st.subheader("🗑️ Clear Faculty Collection")
                
                total_chunks = collection_info.get("count", 0)
                
                if total_chunks > 0:
                    st.warning(f"Found {total_chunks} chunks in {selected_faculty} faculty collection")
                    
                    if st.button(f"🗑️ Clear {selected_faculty.title()} Faculty Collection", type="secondary"):
                        st.session_state[f"confirm_clear_faculty_{selected_faculty}"] = True
                    
                    # Clear confirmation
                    if st.session_state.get(f"confirm_clear_faculty_{selected_faculty}", False):
                        st.error(f"⚠️ **WARNING: This will delete ALL documents from {selected_faculty} faculty!**")
                        st.write("This action cannot be undone")
                        
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Yes, Clear Faculty Collection", type="primary"):
                                if embedding_service.clear_all_documents():
                                    st.success(f"✅ {selected_faculty.title()} faculty collection cleared successfully!")
                                    st.session_state[f"confirm_clear_faculty_{selected_faculty}"] = False
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to clear collection")
                        
                        with col_cancel:
                            if st.button("❌ Cancel"):
                                st.session_state[f"confirm_clear_faculty_{selected_faculty}"] = False
                                st.rerun()
                else:
                    st.info(f"{selected_faculty.title()} faculty collection is empty - no documents to clear")
            
            else:
                st.error(f"Error getting statistics: {stats['error']}")
        
    except Exception as e:
        st.error(f"Error initializing document management: {e}")
        st.write("Please check your configuration and try again.")

def show_faculty_analytics():
    """Show detailed faculty analytics"""
    st.header("📈 Faculty Analytics")
    
    db = get_db()
    
    try:
        # Faculty overview
        st.subheader("🎓 Faculty Overview")
        
        # Get faculty statistics
        faculty_users = db.query(User.faculty, func.count(User.id).label('user_count')).group_by(User.faculty).all()
        faculty_tokens = db.query(User.faculty, func.sum(User.tokens_used).label('total_tokens')).group_by(User.faculty).all()
        
        # Document collections
        collections = EmbeddingService.get_all_faculty_collections()
        
        # Combine data
        faculty_stats = {}
        
        # Initialize with user data
        for faculty, user_count in faculty_users:
            faculty_name = faculty or 'general'
            faculty_stats[faculty_name] = {
                'users': user_count,
                'tokens': 0,
                'documents': 0
            }
        
        # Add token data
        for faculty, token_count in faculty_tokens:
            faculty_name = faculty or 'general'
            if faculty_name in faculty_stats:
                faculty_stats[faculty_name]['tokens'] = token_count or 0
        
        # Add document data
        for collection in collections:
            faculty_name = collection['faculty']
            if faculty_name in faculty_stats:
                faculty_stats[faculty_name]['documents'] = collection['count']
            else:
                faculty_stats[faculty_name] = {
                    'users': 0,
                    'tokens': 0,
                    'documents': collection['count']
                }
        
        # Display as table
        if faculty_stats:
            faculty_df = pd.DataFrame.from_dict(faculty_stats, orient='index')
            faculty_df.index.name = 'Faculty'
            faculty_df.columns = ['Users', 'Tokens Used', 'Documents']
            faculty_df = faculty_df.reset_index()
            
            st.dataframe(faculty_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(faculty_df, x='Faculty', y=['Users', 'Documents'],
                           title='Users and Documents by Faculty',
                           barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(faculty_df, x='Faculty', y='Tokens Used',
                           title='Token Usage by Faculty')
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity by faculty
        st.subheader("📊 Recent Activity (Last 7 Days)")
        
        week_ago = datetime.now() - timedelta(days=7)
        recent_chats = db.query(
            User.faculty,
            func.count(ChatLog.id).label('chat_count'),
            func.sum(ChatLog.tokens_used).label('token_usage')
        ).join(User, ChatLog.user_id == User.id).filter(
            ChatLog.timestamp >= week_ago
        ).group_by(User.faculty).all()
        
        if recent_chats:
            recent_df = pd.DataFrame(recent_chats, columns=['Faculty', 'Chats', 'Tokens'])
            recent_df['Faculty'] = recent_df['Faculty'].fillna('general')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(recent_df, values='Chats', names='Faculty',
                           title='Chat Distribution (Last 7 Days)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(recent_df, x='Faculty', y='Tokens',
                           title='Token Usage (Last 7 Days)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent activity found")
    
    except Exception as e:
        st.error(f"Error loading faculty analytics: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()