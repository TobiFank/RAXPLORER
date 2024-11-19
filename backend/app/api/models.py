# app/api/models.py
from fastapi import APIRouter, Depends
from ..dependencies import get_model_config_service
from ..schemas.model import ModelConfig
from ..services.llm import LLMService
from ..services.model_config import ModelConfigService

router = APIRouter(prefix="/model")

@router.get("/config")
async def get_config(
        service: ModelConfigService = Depends(get_model_config_service)
) -> list[ModelConfig]:
    return await service.get_configs()

@router.post("/config")
async def save_config(
        config: ModelConfig,
        service: ModelConfigService = Depends(get_model_config_service)
):
    await service.save_config(config)

@router.post("/validate")
async def validate_config(config: ModelConfig) -> dict:
    return await LLMService.validate_config(config)