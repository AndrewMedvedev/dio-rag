from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    USER = "user"
    AI = "ai"


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Role
    text: str

    model_config = ConfigDict(from_attributes=True)
