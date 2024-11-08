# app/utils/errors.py
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException


class ErrorCode:
    # Chat related errors
    CHAT_001 = "CHAT_001"  # Chat not found
    CHAT_002 = "CHAT_002"  # Invalid chat operation

    # File related errors
    FILE_001 = "FILE_001"  # File upload failed
    FILE_002 = "FILE_002"  # Unsupported file type
    FILE_003 = "FILE_003"  # File processing failed

    # Model related errors
    MODEL_001 = "MODEL_001"  # Invalid model configuration
    MODEL_002 = "MODEL_002"  # API key validation failed

    # RAG related errors
    RAG_001 = "RAG_001"  # Vector storage error
    RAG_002 = "RAG_002"  # Embedding generation failed
    RAG_003 = "RAG_003"  # Document processing error


class APIError(HTTPException):
    def __init__(
            self,
            code: str,
            message: str,
            status_code: int = 400,
            details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = code
        self.error_message = message
        self.error_details = details
        super().__init__(status_code=status_code, detail=message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.error_message,
                "details": self.error_details
            },
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }


class ChatError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.CHAT_001, message, 404, details)


class ModelConfigError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.MODEL_001, message, 400, details)


class FileProcessingError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.FILE_001, message, 500, details)


class VectorStoreError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.RAG_001, message, 500, details)


class EmbeddingError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.RAG_002, message, 500, details)


class DocumentProcessingError(APIError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.RAG_003, message, 500, details)
