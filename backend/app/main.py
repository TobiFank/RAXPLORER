# app/main.py
from app.api.v1 import chat, files, models  # Add models import
from app.core.config import settings
from app.core.middleware import ErrorHandlingMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling middleware
app.add_middleware(ErrorHandlingMiddleware)

# Include routers
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat")
app.include_router(files.router, prefix=f"{settings.API_V1_STR}/files")
app.include_router(models.router, prefix=f"{settings.API_V1_STR}/model")  # Add models router


@app.get("/")
async def root():
    return {"message": "Welcome to Chat RAG Backend API"}
