# app/services/model_config.py
from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime

from app.models.model_config import ModelConfig
from app.schemas.model_config import ModelConfigCreate
from app.services.llm.base import BaseLLMService
from app.services.llm.factory import create_llm_service
from app.utils.errors import ModelConfigError

class ModelConfigService:
    def __init__(self, db: Session):
        self.db = db

    async def validate_config(self, config: ModelConfigCreate) -> tuple[bool, Optional[List[str]]]:
        """Validate model configuration including API key testing"""
        issues = []

        try:
            # Create temporary LLM service to test configuration
            llm_service = await create_llm_service(
                provider=config.provider,
                api_key=config.api_key,
                model=config.model
            )

            # Test connection/authentication
            is_healthy = await llm_service.health_check()
            if not is_healthy:
                issues.append(f"Failed to connect to {config.provider} service")

        except Exception as e:
            issues.append(f"Configuration validation failed: {str(e)}")

        return (len(issues) == 0, issues if issues else None)

    def save_config(self, config: ModelConfigCreate) -> ModelConfig:
        """Save or update model configuration"""
        # Check for existing config
        db_config = self.db.query(ModelConfig).filter(
            ModelConfig.provider == config.provider
        ).first()

        if db_config:
            # Update existing config
            for key, value in config.dict(exclude_unset=True).items():
                setattr(db_config, key, value)
            db_config.updated_at = datetime.utcnow()
        else:
            # Create new config
            db_config = ModelConfig(
                id=config.provider,
                **config.dict()
            )
            self.db.add(db_config)

        try:
            self.db.commit()
            self.db.refresh(db_config)
            return db_config
        except Exception as e:
            self.db.rollback()
            raise ModelConfigError(f"Failed to save configuration: {str(e)}")

    def get_config(self, provider: str) -> Optional[ModelConfig]:
        """Get configuration for a specific provider"""
        return self.db.query(ModelConfig).filter(
            ModelConfig.provider == provider
        ).first()

    def list_configs(self) -> List[ModelConfig]:
        """Get all model configurations"""
        return self.db.query(ModelConfig).all()