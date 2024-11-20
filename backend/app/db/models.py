# app/db/models.py
from sqlalchemy import Column, String, DateTime, Integer, JSON, Float
from sqlalchemy.sql import func
from uuid import uuid4

from .session import Base

class ChatModel(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    title = Column(String, nullable=False)
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, server_default=func.now())

class FileModel(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    size = Column(String, nullable=False)
    pages = Column(Integer, nullable=False)
    vector_store_id = Column(String, nullable=False)
    uploaded_at = Column(DateTime, server_default=func.now())
    embedding_provider = Column(String, nullable=False)

class ModelConfigModel(Base):
    __tablename__ = "model_configs"

    provider = Column(String, primary_key=True)
    model = Column(String, nullable=False)
    temperature = Column(Float, nullable=False)
    system_message = Column(String)
    api_key = Column(String)