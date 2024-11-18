# app/api/files.py
from fastapi import APIRouter, UploadFile, Depends

from ..dependencies import get_storage_service
from ..schemas.file import FileMetadata
from ..services.storage import StorageService

router = APIRouter(prefix="/files")


@router.post("/upload")
async def upload_file(
        file: UploadFile,
        storage_service: StorageService = Depends(get_storage_service)
) -> FileMetadata:
    return await storage_service.upload(file)


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
