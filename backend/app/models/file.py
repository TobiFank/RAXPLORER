# app/models/file.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from app.core.database import Base

class File(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    size = Column(Integer, nullable=False)  # Size in bytes
    pages = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    vectorized = Column(Boolean, default=False)

    # Relationship with chunks
    chunks = relationship("FileChunk", back_populates="file", cascade="all, delete-orphan")

class FileChunk(Base):
    __tablename__ = "file_chunks"

    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    vector_id = Column(String)  # Reference to vector in vector store

    # Relationship with file
    file = relationship("File", back_populates="chunks")