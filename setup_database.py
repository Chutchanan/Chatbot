import os
import sys
import shutil
import chromadb
from datetime import datetime

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import the modules
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# Configuration
try:
    from config import DATABASE_URL, DEFAULT_TOKEN_LIMIT, CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, FACULTIES
except ImportError:
    DATABASE_URL = "sqlite:///./chatbot.db"
    DEFAULT_TOKEN_LIMIT = 1000
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "company_documents"
    FACULTIES = ["engineering", "business", "science", "arts"]

# Database paths
DB_PATH = "./chatbot.db"

# Create Base and Models directly here to avoid import issues
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

def clean_everything():
    """Completely clean all existing data"""
    print("🧹 COMPLETE CLEANUP - Removing all existing data...")
    
    # Remove SQLite database
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"🗑️ Removed database file: {DB_PATH}")
        except Exception as e:
            print(f"❌ Failed to remove database: {e}")
            return False
    
    # Remove ChromaDB directory
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
            print(f"🗑️ Removed ChromaDB directory: {CHROMA_PERSIST_DIRECTORY}")
        except Exception as e:
            print(f"❌ Failed to remove ChromaDB directory: {e}")
            return False
    
    # Remove any temporary files
    temp_files = [f for f in os.listdir('.') if f.startswith('temp_')]
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"🗑️ Removed temp file: {temp_file}")
        except:
            pass
    
    print("✅ Complete cleanup finished!")
    return True

def create_database_tables():
    """Create database tables"""
    print("📊 Creating database tables...")
    
    try:
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        return engine
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        return None

def create_faculty_users():
    """Create users for each faculty + admin users"""
    print("👥 Creating faculty users...")
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Create users for each faculty
        users_to_create = [
            # Faculty-specific users
            {"username": "engineering_user", "email": "eng@university.edu", "faculty": "engineering", "token_limit": 2000},
            {"username": "business_user", "email": "biz@university.edu", "faculty": "business", "token_limit": 1500},
            {"username": "science_user", "email": "sci@university.edu", "faculty": "science", "token_limit": 2500},
            {"username": "arts_user", "email": "arts@university.edu", "faculty": "arts", "token_limit": 1200},
            
            # General users
            {"username": "demo", "email": "demo@university.edu", "faculty": "general", "token_limit": 1000},
            {"username": "user1", "email": "user1@university.edu", "faculty": "general", "token_limit": 1000},
            {"username": "user2", "email": "user2@university.edu", "faculty": "general", "token_limit": 1000},
            
            # Admin test user
            {"username": "admin_user", "email": "admin@university.edu", "faculty": "general", "token_limit": 5000},
        ]
        
        created_count = 0
        for user_data in users_to_create:
            existing_user = db.query(User).filter(User.username == user_data["username"]).first()
            if not existing_user:
                new_user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    faculty=user_data["faculty"],
                    token_limit=user_data["token_limit"],
                    tokens_used=0,
                    is_active=True
                )
                db.add(new_user)
                print(f"✅ Created user: {user_data['username']} ({user_data['faculty']} faculty)")
                created_count += 1
            else:
                print(f"ℹ️ User already exists: {user_data['username']}")
        
        db.commit()
        print(f"✅ Created {created_count} new users!")
        
        # Show summary
        print("\n📊 User Summary:")
        users = db.query(User).all()
        faculty_counts = {}
        for user in users:
            faculty = user.faculty or 'general'
            faculty_counts[faculty] = faculty_counts.get(faculty, 0) + 1
        
        for faculty, count in faculty_counts.items():
            print(f"  🎓 {faculty.title()}: {count} users")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating users: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def create_clean_chroma_collections():
    """Create clean ChromaDB collections"""
    print("📚 Creating clean ChromaDB collections...")
    
    try:
        # Create ChromaDB directory
        os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Create general collection
        try:
            general_collection = client.create_collection(COLLECTION_NAME)
            print(f"✅ Created general collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ General collection already exists: {COLLECTION_NAME}")
            else:
                print(f"❌ Failed to create general collection: {e}")
                return False
        
        # Create faculty-specific collections
        for faculty in FACULTIES:
            collection_name = f"{COLLECTION_NAME}_{faculty}"
            try:
                faculty_collection = client.create_collection(collection_name)
                print(f"✅ Created faculty collection: {collection_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"ℹ️ Faculty collection already exists: {collection_name}")
                else:
                    print(f"❌ Failed to create faculty collection {collection_name}: {e}")
        
        # Verify collections
        print("\n📋 Verified Collections:")
        all_collections = client.list_collections()
        for collection in all_collections:
            print(f"  📚 {collection.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating ChromaDB collections: {e}")
        return False

def create_sample_documents():
    """Create sample documents in data directory"""
    print("📄 Creating sample documents...")
    
    # Create data directory structure
    data_dirs = {
        "data/general": "General university information",
        "data/engineering": "Engineering faculty documents", 
        "data/business": "Business faculty documents",
        "data/science": "Science faculty documents",
        "data/arts": "Arts faculty documents"
    }
    
    sample_docs = {
        "data/general/university_overview.txt": """
University Overview

Welcome to our prestigious university! We are a leading institution of higher education 
committed to excellence in teaching, research, and service.

Founded in 1950, we serve over 25,000 students across multiple faculties including 
Engineering, Business, Science, and Arts.

Our campus features state-of-the-art facilities, world-class libraries, and 
cutting-edge research centers.

Contact Information:
- Main Office: (555) 123-4567
- Email: info@university.edu
- Address: 123 University Ave, Academic City
        """,
        
        "data/general/policies.txt": """
University Policies and Guidelines

Student Code of Conduct:
1. Academic integrity is paramount
2. Respect for all community members
3. Compliance with campus regulations

Academic Policies:
- Minimum GPA requirement: 2.0
- Maximum course load: 18 credits per semester
- Attendance policy: 75% minimum attendance required

Support Services:
- Academic advising available
- Counseling and psychological services
- Career development center
- Library and research support
        """,
        
        "data/engineering/technical_specs.txt": """
Engineering Faculty - Technical Specifications

Our Engineering programs offer cutting-edge education in:
- Computer Science and Software Engineering
- Electrical and Electronic Engineering  
- Mechanical Engineering
- Civil Engineering
- Biomedical Engineering

Laboratory Facilities:
- Advanced robotics lab
- High-performance computing cluster
- Materials testing laboratory
- CAD/CAM workshop

Research Areas:
- Artificial Intelligence and Machine Learning
- Renewable Energy Systems
- Structural Engineering
- Biomedical Device Development
        """,
        
        "data/business/business_programs.txt": """
Business Faculty Programs

Our Business School offers comprehensive programs in:
- Business Administration (MBA)
- Finance and Investment
- Marketing and Digital Commerce
- Human Resources Management
- Entrepreneurship and Innovation

Special Features:
- Industry partnerships with Fortune 500 companies
- Internship placement program
- Business incubator for startups
- Executive education programs

Career Services:
- Job placement assistance
- Networking events
- Alumni mentorship program
- Professional development workshops
        """,
        
        "data/science/research_overview.txt": """
Science Faculty Research Overview

Our Science Faculty conducts world-class research in:
- Biology and Life Sciences
- Chemistry and Biochemistry  
- Physics and Astronomy
- Mathematics and Statistics
- Environmental Science

Major Research Facilities:
- Molecular biology laboratory
- Observatory and planetarium
- Greenhouse and botanical garden
- High-energy physics lab

Current Research Projects:
- Climate change impact studies
- Drug discovery and development
- Quantum computing research
- Biodiversity conservation
        """,
        
        "data/arts/arts_programs.txt": """
Arts Faculty Programs and Resources

Our Arts Faculty offers diverse programs in:
- Fine Arts and Visual Design
- Music and Performing Arts
- Literature and Creative Writing
- History and Cultural Studies
- Philosophy and Ethics

Facilities:
- Art studios and exhibition spaces
- Concert hall and performance theater
- Digital media laboratory
- Archives and special collections

Cultural Events:
- Annual arts festival
- Student exhibitions
- Guest artist residencies
- Community outreach programs
        """
    }
    
    try:
        created_count = 0
        for file_path, content in sample_docs.items():
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            print(f"✅ Created: {file_path}")
            created_count += 1
        
        print(f"✅ Created {created_count} sample documents!")
        
        # Show directory structure
        print("\n📁 Document Structure:")
        for data_dir, description in data_dirs.items():
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                print(f"  📂 {data_dir}: {len(files)} files ({description})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample documents: {e}")
        return False

def verify_setup():
    """Verify that everything was set up correctly"""
    print("\n🔍 Verifying setup...")
    
    try:
        # Check database
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        user_count = db.query(User).count()
        chat_count = db.query(ChatLog).count()
        
        print(f"✅ Database: {user_count} users, {chat_count} chat logs")
        
        # Check ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collections = client.list_collections()
        print(f"✅ ChromaDB: {len(collections)} collections")
        
        for collection in collections:
            print(f"  📚 {collection.name}: {collection.count()} documents")
        
        # Check sample documents
        doc_count = 0
        for root, dirs, files in os.walk("data"):
            doc_count += len([f for f in files if f.endswith(('.txt', '.pdf', '.csv'))])
        
        print(f"✅ Sample documents: {doc_count} files")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🎓 COMPLETE FACULTY CHATBOT SETUP")
    print("=" * 50)
    print("This will completely clean and rebuild everything!")
    print("- SQLite database")
    print("- ChromaDB collections") 
    print("- Sample documents")
    print("- User accounts")
    print("=" * 50)
    
    confirm = input("⚠️ This will DELETE ALL existing data. Continue? (y/N): ")
    
    if confirm.lower() not in ['y', 'yes']:
        print("❌ Setup cancelled")
        sys.exit(0)
    
    print("\n🚀 Starting complete setup...")
    
    # Step 1: Clean everything
    if not clean_everything():
        print("❌ Cleanup failed")
        sys.exit(1)
    
    # Step 2: Create database tables
    if not create_database_tables():
        print("❌ Database creation failed") 
        sys.exit(1)
    
    # Step 3: Create users
    if not create_faculty_users():
        print("❌ User creation failed")
        sys.exit(1)
    
    # Step 4: Create ChromaDB collections
    if not create_clean_chroma_collections():
        print("❌ ChromaDB setup failed")
        sys.exit(1)
    
    # Step 5: Create sample documents
    if not create_sample_documents():
        print("❌ Sample document creation failed")
        sys.exit(1)
    
    # Step 6: Verify setup
    if not verify_setup():
        print("❌ Setup verification failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    print("\n👤 Test Users Created:")
    print("Faculty Users:")
    print("  - engineering_user/engineering_user (Engineering, 2000 tokens)")
    print("  - business_user/business_user (Business, 1500 tokens)")
    print("  - science_user/science_user (Science, 2500 tokens)")
    print("  - arts_user/arts_user (Arts, 1200 tokens)")
    print("\nGeneral Users:")
    print("  - demo/demo (General, 1000 tokens)")
    print("  - user1/user1 (General, 1000 tokens)")
    print("  - user2/user2 (General, 1000 tokens)")
    print("  - admin_user/admin_user (General, 5000 tokens)")
    
    print("\n📚 Collections Created:")
    print("  - company_documents (general)")
    print("  - company_documents_engineering")
    print("  - company_documents_business")
    print("  - company_documents_science")
    print("  - company_documents_arts")
    
    print("\n📄 Sample Documents:")
    print("  - data/general/ (university info)")
    print("  - data/engineering/ (technical specs)")
    print("  - data/business/ (business programs)")
    print("  - data/science/ (research info)")
    print("  - data/arts/ (arts programs)")
    
    print("\n🚀 Next Steps:")
    print("1. Run admin console: python run_admin.py")
    print("2. Upload documents to faculty collections")
    print("3. Test chatbot: python run_chatbot.py")
    print("4. Login with faculty users to test dual RAG access")
    
    print("\n✨ Your faculty chatbot system is ready!")