# app/core/middleware.py
import logging
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
            logging.error(f"API Error: {str(e)}")  # Add logging
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
        except SQLAlchemyError as e:
            logging.error(f"Database Error: {str(e)}")  # Add logging
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
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")  # Add logging
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
