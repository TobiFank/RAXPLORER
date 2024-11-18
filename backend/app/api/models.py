# app/api/models.py
from fastapi import APIRouter
from ..schemas.model import ModelConfig
from ..services.llm import LLMService

router = APIRouter(prefix="/model")

@router.post("/validate")
async def validate_config(config: ModelConfig) -> dict:
    return await LLMService.validate_config(config)

@router.post("/config")
async def save_config(config: ModelConfig):
    await LLMService.save_config(config)

@router.get("/config")
async def get_config() -> list[ModelConfig]:
    return await LLMService.get_configs()