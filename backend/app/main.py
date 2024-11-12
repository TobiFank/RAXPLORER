# app/main.py
import logging

from app.api.v1 import chat, files, models  # Add models import
from app.core.config import settings
from app.core.database import verify_db_connection
from app.core.middleware import ErrorHandlingMiddleware
from app.utils.vector_store import MilvusVectorStore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling middleware
app.add_middleware(ErrorHandlingMiddleware)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Verify database
        await verify_db_connection()

        # Verify Milvus connection
        vector_store = MilvusVectorStore(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.MILVUS_COLLECTION,
        )
        await vector_store.initialize()

        logging.info("All services initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise


# Include routers
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat")
app.include_router(files.router, prefix=f"{settings.API_V1_STR}/files")
app.include_router(models.router, prefix=f"{settings.API_V1_STR}/model")  # Add models router


@app.get("/")
async def root():
    return {"message": "Welcome to Chat RAG Backend API"}
