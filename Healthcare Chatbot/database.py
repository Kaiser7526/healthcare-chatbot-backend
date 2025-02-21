from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ✅ Database URL (SQLite)
DATABASE_URL = "sqlite:///./chatbot.db"  # Use "postgresql://user:password@localhost/dbname" for PostgreSQL

# ✅ Create Database Engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# ✅ Create a Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Define Base Class for Models
Base = declarative_base()
