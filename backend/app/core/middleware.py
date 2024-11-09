# app/core/middleware.py
from datetime import datetime
from http.client import HTTPException

from app.utils.errors import APIError
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except APIError as e:
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred",
                        "details": str(e) if not isinstance(e, HTTPException) else None
                    },
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except SQLAlchemyError as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "DATABASE_ERROR",
                        "message": "Database operation failed",
                        "details": str(e)
                    },
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
