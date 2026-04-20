from fastapi import APIRouter, status

from ..chatbot import call_chatbot
from ..schemas import Message, Role

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    path="/completions",
    status_code=status.HTTP_200_OK,
    response_model=Message,
    summary="Чат с RAG агентом",
)
async def create_chat_completion(user_message: Message) -> Message:
    response = await call_chatbot(
        user_id=user_message.id, user_prompt=user_message.text, chat_id=user_message.chat_id
    )

    return Message(chat_id=user_message.chat_id, role=Role.AI, text=response)
