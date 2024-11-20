# app/api/chat.py
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..dependencies import get_chat_service
from ..schemas.chat import Chat, MessageRequest
from ..services.chat import ChatService

router = APIRouter(prefix="/chat")


@router.get("/chats")
async def get_chats(
        chat_service: ChatService = Depends(get_chat_service)
) -> list[Chat]:
    return await chat_service.get_chats()


@router.get("/chats/{chat_id}")
async def get_chat(
        chat_id: str,
        chat_service: ChatService = Depends(get_chat_service)
) -> Chat:
    return await chat_service.get_chat(chat_id)


@router.delete("/chats/{chat_id}")
async def delete_chat(
        chat_id: str,
        chat_service: ChatService = Depends(get_chat_service)
):
    await chat_service.delete_chat(chat_id)


@router.patch("/chats/{chat_id}")
async def update_chat_title(
        chat_id: str,
        title: str,
        chat_service: ChatService = Depends(get_chat_service)
) -> Chat:
    return await chat_service.update_title(chat_id, title)


@router.post("/chats/{chat_id}/messages")
async def send_message(
        chat_id: str,
        request: MessageRequest,
        chat_service: ChatService = Depends(get_chat_service)
) -> StreamingResponse:
    return StreamingResponse(
        chat_service.stream_response(
            chat_id,
            request.content,
            request.modelConfig
        ),
        media_type='text/plain'
    )

@router.post("/chats")
async def create_chat(
        chat_service: ChatService = Depends(get_chat_service)
) -> Chat:
    return await chat_service.create_chat()
