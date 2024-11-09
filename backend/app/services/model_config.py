# app/services/model_config.py
from datetime import datetime
from typing import Optional, List, Tuple

import httpx
from app.core.config import settings
from app.models.model_config import ModelConfig
from app.schemas.model_config import ModelConfigCreate
from app.services.llm.factory import create_llm_service
from app.utils.errors import ModelConfigError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


class ModelConfigService:
    def __init__(self, db: Session):
        self.db = db

    async def _check_ollama_status(self, client: httpx.AsyncClient) -> Tuple[bool, Optional[str]]:
        """Check if Ollama service is running and accessible"""
        try:
            response = await client.get("/api/tags")
            if response.status_code == 200:
                return True, None
            return False, f"Ollama service returned status code: {response.status_code}"
        except httpx.ConnectError:
            return False, "Cannot connect to Ollama service. Please ensure Ollama is running."
        except Exception as e:
            return False, f"Error checking Ollama status: {str(e)}"

    async def _load_ollama_model(self, client: httpx.AsyncClient, model_name: str) -> Tuple[bool, Optional[str]]:
        """Load or pull an Ollama model"""
        try:
            # First check if model exists
            show_response = await client.get(f"/api/show?name={model_name}")

            if show_response.status_code == 404:
                # Model doesn't exist, try to pull it
                pull_response = await client.post(
                    "/api/pull",
                    json={"name": model_name},
                    timeout=60.0
                )
                if pull_response.status_code != 200:
                    return False, f"Failed to pull model {model_name}: {pull_response.text}"

            return True, None

        except httpx.TimeoutException:
            return False, f"Timeout while loading model {model_name}. The model may be too large or the server may be busy."
        except Exception as e:
            return False, f"Error loading model {model_name}: {str(e)}"

    async def validate_config(self, config: ModelConfigCreate) -> Tuple[bool, Optional[List[str]]]:
        """Validate model configuration including API key testing and Ollama model verification"""
        issues = []

        try:
            if config.provider == 'ollama':
                model_name = config.ollamaModel or config.model
                if not model_name:
                    issues.append("Ollama model name is required")
                    return False, issues

                async with httpx.AsyncClient(
                        base_url=settings.OLLAMA_HOST,
                        timeout=30.0
                ) as client:
                    # Check Ollama service status
                    is_running, status_error = await self._check_ollama_status(client)
                    if not is_running:
                        issues.append(status_error)
                        return False, issues

                    # Try to load the model
                    model_loaded, load_error = await self._load_ollama_model(
                        client,
                        model_name
                    )
                    if not model_loaded:
                        issues.append(load_error)
                        return False, issues

            elif config.provider in ['claude', 'chatgpt']:
                if not config.api_key:
                    issues.append(f"{config.provider} requires an API key")
                else:
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
            # For Ollama, ensure we have a valid model name and set it correctly
            if config.provider == 'ollama':
                if not config.ollamaModel:
                    raise ModelConfigError("Ollama model name is required")
                config.model = config.ollamaModel  # Set model to ollamaModel value

            # Validate configuration
            is_valid, issues = await self.validate_config(config)
            if not is_valid:
                raise ModelConfigError("Invalid configuration", {"issues": issues})

            # Save to database
            db_config = self.db.query(ModelConfig).filter(
                ModelConfig.provider == config.provider
            ).first()

            config_dict = config.dict(exclude_unset=True)
            if db_config:
                for key, value in config_dict.items():
                    setattr(db_config, key, value)
                db_config.updated_at = datetime.utcnow()
            else:
                db_config = ModelConfig(
                    id=config.provider,
                    **config_dict
                )
                self.db.add(db_config)

            try:
                self.db.commit()
                self.db.refresh(db_config)
                return db_config
            except SQLAlchemyError as e:
                self.db.rollback()
                raise ModelConfigError(f"Database error: {str(e)}")

        except Exception as e:
            self.db.rollback()
            if isinstance(e, ModelConfigError):
                raise e
            raise ModelConfigError(f"Failed to save configuration: {str(e)}")

    def get_config(self, provider: str) -> Optional[ModelConfig]:
        """Get configuration for a specific provider"""
        return self.db.query(ModelConfig).filter(
            ModelConfig.provider == provider
        ).first()

    def list_configs(self) -> List[ModelConfig]:
        """Get all model configurations"""
        return self.db.query(ModelConfig).all()
