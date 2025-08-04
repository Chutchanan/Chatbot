import os
import sys
import shutil
import sqlite3
import chromadb
from datetime import datetime

def force_cleanup():
    """Force cleanup everything"""
    print("🔥 FORCE CLEANUP - Removing everything...")
    
    files_to_remove = [
        "./chatbot.db",
        "./chatbot.db-journal",  # SQLite journal file
        "./chatbot.db-wal",      # SQLite WAL file
        "./chatbot.db-shm",      # SQLite shared memory file
    ]
    
    dirs_to_remove = [
        "./chroma_db",
        "./__pycache__",
        "./database/__pycache__",
        "./services/__pycache__",
        "./frontend/__pycache__",
    ]
    
    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"🗑️ Removed directory: {dir_path}")
            except Exception as e:
                print(f"❌ Failed to remove {dir_path}: {e}")
    
    # Remove temp files
    for file in os.listdir('.'):
        if file.startswith('temp_'):
            try:
                os.remove(file)
                print(f"🗑️ Removed temp file: {file}")
            except:
                pass
    
    print("✅ Force cleanup completed!")

def create_fresh_database():
    """Create completely fresh database"""
    print("📊 Creating fresh database...")
    
    db_path = "./chatbot.db"
    
    # Make sure database doesn't exist
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database with raw SQL to avoid model conflicts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table with correct schema
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            faculty TEXT DEFAULT 'general',
            token_limit INTEGER DEFAULT 1000,
            tokens_used INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create chat_logs table
    cursor.execute("""
        CREATE TABLE chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query TEXT,
            response TEXT,
            tokens_used INTEGER,
            has_file_upload BOOLEAN DEFAULT 0,
            file_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_users_username ON users(username)")
    cursor.execute("CREATE INDEX idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX idx_chat_logs_user_id ON chat_logs(user_id)")
    cursor.execute("CREATE INDEX idx_chat_logs_timestamp ON chat_logs(timestamp)")
    
    conn.commit()
    conn.close()
    
    print("✅ Fresh database created!")

def create_users():
    """Create users with raw SQL"""
    print("👥 Creating users...")
    
    conn = sqlite3.connect("./chatbot.db")
    cursor = conn.cursor()
    
    users = [
        ("engineering_user", "eng@university.edu", "engineering", 2000),
        ("business_user", "biz@university.edu", "business", 1500),
        ("science_user", "sci@university.edu", "science", 2500),
        ("arts_user", "arts@university.edu", "arts", 1200),
        ("demo", "demo@university.edu", "general", 1000),
        ("user1", "user1@university.edu", "general", 1000),
        ("user2", "user2@university.edu", "general", 1000),
        ("admin_user", "admin@university.edu", "general", 5000),
    ]
    
    for username, email, faculty, token_limit in users:
        try:
            cursor.execute("""
                INSERT INTO users (username, email, faculty, token_limit, tokens_used, is_active)
                VALUES (?, ?, ?, ?, 0, 1)
            """, (username, email, faculty, token_limit))
            print(f"✅ Created user: {username} ({faculty})")
        except sqlite3.IntegrityError:
            print(f"ℹ️ User already exists: {username}")
    
    conn.commit()
    conn.close()
    print("✅ Users created!")

def create_chroma_collections():
    """Create clean ChromaDB collections"""
    print("📚 Creating ChromaDB collections...")
    
    chroma_dir = "./chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)
    
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        
        # Collection names
        collections = [
            "company_documents",           # general
            "company_documents_engineering",
            "company_documents_business", 
            "company_documents_science",
            "company_documents_arts"
        ]
        
        for collection_name in collections:
            try:
                client.create_collection(collection_name)
                print(f"✅ Created collection: {collection_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"ℹ️ Collection exists: {collection_name}")
                else:
                    print(f"❌ Failed to create {collection_name}: {e}")
        
        print("✅ ChromaDB collections ready!")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB setup failed: {e}")
        return False

def create_sample_data():
    """Create sample documents"""
    print("📄 Creating sample documents...")
    
    # Create data directories
    data_dirs = [
        "data/general",
        "data/engineering", 
        "data/business",
        "data/science",
        "data/arts"
    ]
    
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Sample documents
    sample_files = {
        "data/general/university_info.txt": """
University Information

Welcome to our university! We offer world-class education across multiple faculties.

General Information:
- Founded: 1950
- Students: 25,000+
- Faculty: Engineering, Business, Science, Arts
- Campus: Modern facilities with latest technology

Contact:
- Phone: (555) 123-4567
- Email: info@university.edu
- Website: www.university.edu

Services:
- Library and research facilities
- Student counseling
- Career services
- Campus health center
        """,
        
        "data/engineering/tech_info.txt": """
Engineering Faculty

Our Engineering department offers cutting-edge programs in:
- Computer Science and AI
- Electrical Engineering
- Mechanical Engineering
- Civil Engineering

Facilities:
- Advanced computer labs
- Robotics workshop
- Materials testing lab
- Design studios

Research Areas:
- Artificial Intelligence
- Renewable Energy
- Smart Cities
- Biomedical Engineering
        """,
        
        "data/business/business_info.txt": """
Business Faculty

Our Business School provides excellent education in:
- Business Administration
- Finance and Accounting
- Marketing and Sales
- Human Resources
- Entrepreneurship

Programs:
- MBA (Master of Business Administration)
- Executive Education
- Professional Certificates
- Industry Partnerships

Career Support:
- Job placement services
- Networking events
- Mentorship programs
- Business incubator
        """,
        
        "data/science/science_info.txt": """
Science Faculty

Our Science department excels in research and education:
- Biology and Life Sciences
- Chemistry and Biochemistry
- Physics and Astronomy
- Mathematics and Statistics
- Environmental Science

Research Facilities:
- Modern laboratories
- Observatory
- Greenhouse
- Field research stations

Current Projects:
- Climate change studies
- Drug discovery
- Space exploration
- Marine biology research
        """,
        
        "data/arts/arts_info.txt": """
Arts Faculty

Our Arts Faculty celebrates creativity and culture:
- Fine Arts and Design
- Music and Performance
- Literature and Writing
- History and Philosophy
- Cultural Studies

Facilities:
- Art studios and galleries
- Concert hall
- Theater
- Digital media lab
- Archives

Programs:
- Bachelor of Fine Arts
- Creative Writing Workshop
- Music Performance
- Art History
- Philosophy seminars
        """
    }
    
    created_count = 0
    for file_path, content in sample_files.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"✅ Created: {file_path}")
            created_count += 1
        except Exception as e:
            print(f"❌ Failed to create {file_path}: {e}")
    
    print(f"✅ Created {created_count} sample documents!")

def verify_setup():
    """Verify the setup"""
    print("\n🔍 Verifying setup...")
    
    # Check database
    try:
        conn = sqlite3.connect("./chatbot.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT faculty, COUNT(*) FROM users GROUP BY faculty")
        faculty_counts = cursor.fetchall()
        
        print(f"✅ Database: {user_count} users")
        for faculty, count in faculty_counts:
            print(f"  🎓 {faculty}: {count} users")
        
        conn.close()
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    
    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        print(f"✅ ChromaDB: {len(collections)} collections")
        for collection in collections:
            print(f"  📚 {collection.name}")
    except Exception as e:
        print(f"❌ ChromaDB check failed: {e}")
        return False
    
    # Check sample files
    doc_count = 0
    for root, dirs, files in os.walk("data"):
        doc_count += len(files)
    print(f"✅ Sample documents: {doc_count} files")
    
    return True

if __name__ == "__main__":
    print("🎓 COMPLETE SYSTEM RESET")
    print("=" * 50)
    print("This will completely rebuild everything from scratch!")
    print("=" * 50)
    
    confirm = input("⚠️ This will DELETE ALL DATA. Continue? (y/N): ")
    
    if confirm.lower() not in ['y', 'yes']:
        print("❌ Reset cancelled")
        sys.exit(0)
    
    print("\n🚀 Starting complete reset...")
    
    # Step 1: Force cleanup
    force_cleanup()
    
    # Step 2: Create fresh database
    create_fresh_database()
    
    # Step 3: Create users
    create_users()
    
    # Step 4: Create ChromaDB collections
    if not create_chroma_collections():
        print("❌ ChromaDB setup failed")
        sys.exit(1)
    
    # Step 5: Create sample documents
    create_sample_data()
    
    # Step 6: Verify
    if not verify_setup():
        print("❌ Verification failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 COMPLETE RESET SUCCESSFUL!")
    print("=" * 50)
    
    print("\n👤 Test Users (password = username):")
    print("  - engineering_user (Engineering, 2000 tokens)")
    print("  - business_user (Business, 1500 tokens)")  
    print("  - science_user (Science, 2500 tokens)")
    print("  - arts_user (Arts, 1200 tokens)")
    print("  - demo (General, 1000 tokens)")
    print("  - user1, user2 (General, 1000 tokens)")
    print("  - admin_user (General, 5000 tokens)")
    
    print("\n📚 Clean Collections:")
    print("  - company_documents (general)")
    print("  - company_documents_engineering")
    print("  - company_documents_business")
    print("  - company_documents_science") 
    print("  - company_documents_arts")
    
    print("\n🚀 Next Steps:")
    print("1. Run admin console: python run_admin.py")
    print("2. Upload documents to faculty collections")
    print("3. Test chatbot: python run_chatbot.py")
    
    print("\n✨ System is ready! No more faculty_id errors!")