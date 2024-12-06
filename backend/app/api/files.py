# app/api/files.py
import json

from fastapi import APIRouter, UploadFile, Depends, Form, File
from pydantic.v1 import parse_raw_as

from ..dependencies import get_storage_service
from ..schemas.file import FileMetadata
from ..schemas.model import ModelConfig
from ..services.storage import StorageService

router = APIRouter(prefix="/files")


@router.post("/")
async def upload_file(
        file: UploadFile = File(...),
        model_config_json: str = Form(...),
        storage_service: StorageService = Depends(get_storage_service)
) -> FileMetadata:
    model_config = ModelConfig(**json.loads(model_config_json))
    return await storage_service.upload(file, model_config)


@router.get("/")
async def get_files(
        storage_service: StorageService = Depends(get_storage_service)
) -> list[FileMetadata]:
    return await storage_service.get_files()


@router.delete("/{file_id}")
async def delete_file(
        file_id: str,
        storage_service: StorageService = Depends(get_storage_service)
):
    await storage_service.delete(file_id)
