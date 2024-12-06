# app/services/model_config.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import ModelConfigModel
from ..schemas.model import ModelConfig


class ModelConfigService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_config(self, config: ModelConfig):
        query = select(ModelConfigModel).filter(ModelConfigModel.provider == config.provider)
        result = await self.db.execute(query)
        model_config = result.scalar_one_or_none()

        config_dict = {
            "provider": config.provider,
            "model": config.model,
            "temperature": config.temperature,
            "system_message": config.systemMessage,
            "api_key": config.apiKey
        }

        if not model_config:
            model_config = ModelConfigModel(**config_dict)
            self.db.add(model_config)
        else:
            for key, value in config_dict.items():
                setattr(model_config, key, value)

        await self.db.commit()

    async def get_configs(self) -> list[ModelConfig]:
        query = select(ModelConfigModel)
        result = await self.db.execute(query)
        configs = result.scalars().all()
        return [ModelConfig(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            systemMessage=config.system_message,
            apiKey=config.api_key
        ) for config in configs]
