# app/api/v1/models.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.model_config import ModelConfigCreate, ModelConfigResponse
from app.services.model_config import ModelConfigService

router = APIRouter()

@router.post("/validate")
async def validate_model_config(
        config: ModelConfigCreate,
        db: Session = Depends(get_db)
):
    service = ModelConfigService(db)
    is_valid, issues = await service.validate_config(config)

    return {
        "valid": is_valid,
        "issues": issues
    }

@router.post("/config", response_model=ModelConfigResponse)
async def save_model_config(
        config: ModelConfigCreate,
        db: Session = Depends(get_db)
):
    service = ModelConfigService(db)

    # Validate first
    is_valid, issues = await service.validate_config(config)
    if not is_valid:
        raise HTTPException(400, f"Invalid configuration: {', '.join(issues)}")

    return service.save_config(config)

@router.get("/config/{provider}", response_model=ModelConfigResponse)
async def get_model_config(
        provider: str,
        db: Session = Depends(get_db)
):
    service = ModelConfigService(db)
    config = service.get_config(provider)
    if not config:
        raise HTTPException(404, f"No configuration found for provider: {provider}")
    return config

@router.get("/config", response_model=List[ModelConfigResponse])
async def list_model_configs(db: Session = Depends(get_db)):
    service = ModelConfigService(db)
    return service.list_configs()