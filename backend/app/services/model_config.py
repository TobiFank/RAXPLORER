# app/services/model_config.py
from datetime import datetime
from typing import Optional, List, Tuple

import httpx
from app.core.config import settings
from app.models.model_config import ModelConfig
from app.schemas.model_config import ModelConfigCreate
from app.services.llm.factory import create_llm_service
from app.utils.errors import ModelConfigError
from sqlalchemy.orm import Session


class ModelConfigService:
    def __init__(self, db: Session):
        self.db = db

    async def validate_config(self, config: ModelConfigCreate) -> Tuple[bool, Optional[List[str]]]:
        """Validate model configuration including API key testing and Ollama model verification"""
        issues = []

        try:
            if config.provider == 'ollama':
                # Check if Ollama is running and model exists
                async with httpx.AsyncClient(base_url=settings.OLLAMA_HOST) as client:
                    try:
                        # Check if Ollama service is running
                        await client.get("/api/tags")

                        # Check if specified model exists
                        response = await client.get(f"/api/show?name={config.model}")
                        if response.status_code == 404:
                            issues.append(f"Model '{config.model}' not found in Ollama")

                    except httpx.ConnectError:
                        issues.append("Cannot connect to Ollama service")
                    except Exception as e:
                        issues.append(f"Ollama error: {str(e)}")

            elif config.provider in ['claude', 'chatgpt']:
                if not config.api_key:
                    issues.append(f"{config.provider} requires an API key")
                else:
                    # Test API key validity
                    try:
                        llm_service = await create_llm_service(
                            provider=config.provider,
                            api_key=config.api_key,
                            model=config.model
                        )
                        is_healthy = await llm_service.health_check()
                        if not is_healthy:
                            issues.append(f"Failed to authenticate with {config.provider} API")
                    except Exception as e:
                        issues.append(f"API validation failed: {str(e)}")
            else:
                issues.append(f"Unsupported provider: {config.provider}")

        except Exception as e:
            issues.append(f"Configuration validation failed: {str(e)}")

        return (len(issues) == 0, issues if issues else None)

    async def save_config(self, config: ModelConfigCreate) -> ModelConfig:
        """Save model configuration and handle Ollama model setup"""
        try:
            # Validate first
            is_valid, issues = await self.validate_config(config)
            if not is_valid:
                raise ModelConfigError("Invalid configuration", {"issues": issues})

            # Handle Ollama model setup if needed
            if config.provider == 'ollama' and config.model:
                async with httpx.AsyncClient(base_url=settings.OLLAMA_HOST) as client:
                    try:
                        # Pull/update the model if not exists
                        await client.post("/api/pull", json={"name": config.model})
                    except Exception as e:
                        raise ModelConfigError(f"Failed to setup Ollama model: {str(e)}")

            # Save to database
            db_config = self.db.query(ModelConfig).filter(
                ModelConfig.provider == config.provider
            ).first()

            if db_config:
                for key, value in config.dict(exclude_unset=True).items():
                    setattr(db_config, key, value)
                db_config.updated_at = datetime.utcnow()
            else:
                db_config = ModelConfig(
                    id=config.provider,
                    **config.dict()
                )
                self.db.add(db_config)

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
