import os
import sys
import sqlite3
from datetime import datetime

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Database configuration
DATABASE_URL = "sqlite:///./chatbot.db"
DB_PATH = "./chatbot.db"

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    return column_name in columns

def migrate_database():
    """Migrate database to add faculty support"""
    print("🔄 Starting database migration...")
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("❌ Users table not found. Please run setup_database.py first.")
            return False
        
        # Add faculty column to users table if it doesn't exist
        if not check_column_exists(cursor, 'users', 'faculty'):
            print("➕ Adding 'faculty' column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN faculty TEXT DEFAULT 'general'")
            print("✅ Added 'faculty' column")
        else:
            print("✅ 'faculty' column already exists")
        
        # Check if chat_logs table exists and add file upload columns
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_logs'")
        if cursor.fetchone():
            # Add has_file_upload column if it doesn't exist
            if not check_column_exists(cursor, 'chat_logs', 'has_file_upload'):
                print("➕ Adding 'has_file_upload' column to chat_logs table...")
                cursor.execute("ALTER TABLE chat_logs ADD COLUMN has_file_upload BOOLEAN DEFAULT 0")
                print("✅ Added 'has_file_upload' column")
            else:
                print("✅ 'has_file_upload' column already exists")
            
            # Add file_type column if it doesn't exist
            if not check_column_exists(cursor, 'chat_logs', 'file_type'):
                print("➕ Adding 'file_type' column to chat_logs table...")
                cursor.execute("ALTER TABLE chat_logs ADD COLUMN file_type TEXT")
                print("✅ Added 'file_type' column")
            else:
                print("✅ 'file_type' column already exists")
        else:
            print("⚠️ chat_logs table not found")
        
        # Update existing users to have 'general' faculty if they have NULL
        cursor.execute("UPDATE users SET faculty = 'general' WHERE faculty IS NULL")
        rows_updated = cursor.rowcount
        if rows_updated > 0:
            print(f"✅ Updated {rows_updated} users to have 'general' faculty")
        
        # Commit changes
        conn.commit()
        print("✅ Database migration completed successfully!")
        
        # Show current table structure
        print("\n📋 Current users table structure:")
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        for column in columns:
            print(f"  - {column[1]} ({column[2]})")
        
        print("\n📋 Current chat_logs table structure:")
        cursor.execute("PRAGMA table_info(chat_logs)")
        columns = cursor.fetchall()
        for column in columns:
            print(f"  - {column[1]} ({column[2]})")
        
        # Show sample data
        print("\n👥 Sample users data:")
        cursor.execute("SELECT username, faculty, token_limit, tokens_used FROM users LIMIT 5")
        users = cursor.fetchall()
        for user in users:
            print(f"  - {user[0]}: {user[1]} faculty, {user[2]} tokens ({user[3]} used)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def create_sample_faculty_users():
    """Create sample users with different faculties"""
    print("\n👥 Creating sample faculty users...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Sample users for different faculties
        sample_users = [
            ("engineering_user", "eng@university.edu", "engineering", 1500),
            ("business_user", "biz@university.edu", "business", 1200),
            ("science_user", "sci@university.edu", "science", 2000),
            ("arts_user", "arts@university.edu", "arts", 1000),
        ]
        
        for username, email, faculty, token_limit in sample_users:
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO users (username, email, faculty, token_limit, tokens_used, is_active, created_at)
                    VALUES (?, ?, ?, ?, 0, 1, ?)
                """, (username, email, faculty, token_limit, datetime.utcnow()))
                print(f"✅ Created user: {username} ({faculty} faculty)")
            else:
                print(f"ℹ️ User already exists: {username}")
        
        conn.commit()
        print("✅ Sample faculty users created!")
        
    except Exception as e:
        print(f"❌ Error creating sample users: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def verify_migration():
    """Verify that the migration was successful"""
    print("\n🔍 Verifying migration...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Test faculty column
        cursor.execute("SELECT COUNT(DISTINCT faculty) as faculty_count FROM users")
        faculty_count = cursor.fetchone()[0]
        print(f"✅ Found {faculty_count} different faculties in users table")
        
        # Test faculty distribution
        cursor.execute("SELECT faculty, COUNT(*) as count FROM users GROUP BY faculty")
        faculties = cursor.fetchall()
        print("📊 Faculty distribution:")
        for faculty, count in faculties:
            print(f"  - {faculty or 'NULL'}: {count} users")
        
        # Test file upload columns
        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE has_file_upload IS NOT NULL")
        file_logs = cursor.fetchone()[0]
        print(f"✅ Chat logs table has {file_logs} entries with file upload data")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🎓 Faculty Database Migration")
    print("=" * 40)
    
    # Check if database file exists
    if not os.path.exists(DB_PATH):
        print("❌ Database file not found. Please run setup_database.py first.")
        print("Command: python setup_database.py")
        sys.exit(1)
    
    # Run migration
    success = migrate_database()
    
    if success:
        # Create sample users
        create_sample_faculty_users()
        
        # Verify migration
        verify_migration()
        
        print("\n🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("1. Run the admin console: python run_admin.py")
        print("2. Login with admin credentials")
        print("3. Check the dashboard and user management")
        print("4. Test faculty-specific document uploads")
        
        print("\n👤 Test users created:")
        print("- engineering_user/engineering_user (Engineering faculty)")
        print("- business_user/business_user (Business faculty)")
        print("- science_user/science_user (Science faculty)")
        print("- arts_user/arts_user (Arts faculty)")
        print("- Plus existing users updated to 'general' faculty")
        
    else:
        print("\n❌ Migration failed. Please check the errors above.")
        sys.exit(1)