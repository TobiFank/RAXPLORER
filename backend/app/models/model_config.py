# app/models/model_config.py
from sqlalchemy import Column, String, Float, JSON, DateTime
from datetime import datetime
from app.core.database import Base

class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(String, primary_key=True)  # e.g., "claude", "chatgpt", "ollama"
    api_key = Column(String, nullable=True)
    model = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    temperature = Column(Float, nullable=False, default=0.7)
    extra_params = Column(JSON, nullable=True)
    ollamaModel = Column(String, nullable=True)  # Add this field
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)