__all__ = ("router",)

from fastapi import APIRouter

from .chat import router as chat_router
from .documents import router as documents_router

router = APIRouter()

api_router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health():
    return {"status": "ok"}


api_router.include_router(chat_router)
api_router.include_router(documents_router)

router.include_router(api_router)
