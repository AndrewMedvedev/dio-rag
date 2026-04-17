from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

TIMEZONE = "Asia/Yekaterinburg"
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
CHROMA_PATH = BASE_DIR / ".chroma"
SQLITE_PATH = BASE_DIR / "checkpoint.sqlite"
load_dotenv(ENV_PATH)


class YandexCloudSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YANDEX_CLOUD_")

    folder_id: str = "<FOLDER_ID>"
    api_key: str = "<API_KEY>"


class RAGSettings(BaseSettings):
    chunk_size: int = 1000
    chunk_overlap: int = 20
    system_prompt: str = ""
    max_conversation_history_length: int = 10

    model_config = SettingsConfigDict(env_prefix="RAG_")


class Settings(BaseSettings):
    yandexcloud: YandexCloudSettings = YandexCloudSettings()
    rag: RAGSettings = RAGSettings()


settings: Final[Settings] = Settings()
