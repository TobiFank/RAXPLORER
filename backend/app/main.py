# app/main.py
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import chat, files, models
from core.config import Settings
from dependencies import get_chat_service, get_storage_service
from services.llm import LLMService
from services.rag import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup Services
    settings = Settings()
    llm_service = LLMService()
    rag_service = RAGService(llm_service)

    # Add to app state
    app.state.settings = settings
    app.state.llm_service = llm_service
    app.state.rag_service = rag_service

    # Setup database
    from .db.init_db import init_db
    await init_db()

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": str(exc.detail),
            "issues": [str(exc.detail)] if isinstance(exc.detail, str) else exc.detail.get("issues", []),
            "code": exc.status_code
        }
    )


# Include routers
app.include_router(chat.router, prefix="/api/v1", dependencies=[Depends(get_chat_service)])
app.include_router(files.router, prefix="/api/v1", dependencies=[Depends(get_storage_service)])
app.include_router(models.router, prefix="/api/v1")
