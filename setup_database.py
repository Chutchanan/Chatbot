import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import the modules
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime

# Database configuration
DATABASE_URL = "sqlite:///./chatbot.db"
DEFAULT_TOKEN_LIMIT = 1000

# Create Base and Models directly here to avoid import issues
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

def create_tables():
    """Create database tables"""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")
    return engine

def create_default_users():
    """Create default users for testing"""
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Create test users
        test_users = [
            {"username": "user1", "email": "user1@company.com", "token_limit": DEFAULT_TOKEN_LIMIT},
            {"username": "user2", "email": "user2@company.com", "token_limit": DEFAULT_TOKEN_LIMIT},
            {"username": "demo", "email": "demo@company.com", "token_limit": 500}
        ]
        
        for user_data in test_users:
            existing_user = db.query(User).filter(User.username == user_data["username"]).first()
            if not existing_user:
                new_user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    token_limit=user_data["token_limit"],
                    tokens_used=0,
                    is_active=True
                )
                db.add(new_user)
                print(f"Created user: {user_data['username']}")
            else:
                print(f"User already exists: {user_data['username']}")
        
        db.commit()
        print("Default users created successfully!")
        
    except Exception as e:
        print(f"Error creating users: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Setting up database...")
    try:
        create_tables()
        create_default_users()
        print("Database setup completed!")
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Please check if all required packages are installed:")