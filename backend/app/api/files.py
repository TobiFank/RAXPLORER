# app/api/files.py
from fastapi import APIRouter, UploadFile
from ..schemas.file import FileMetadata
from ..services.storage import StorageService

router = APIRouter(prefix="/files")

@router.post("/upload")
async def upload_file(file: UploadFile) -> FileMetadata:
    return await StorageService.upload(file)

@router.get("/")
async def get_files() -> list[FileMetadata]:
    return await StorageService.get_files()

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    await StorageService.delete(file_id)