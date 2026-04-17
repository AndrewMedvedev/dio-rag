from typing import Final

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import ModelProfile
from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from pydantic import SecretStr

from .settings import settings

PROMPT_SUMMARIZE_CHAT = """
Ты — система управления памятью ассистента. Проанализируй диалог и создай структурированное резюме для продолжения разговора.

**Важно**: весь твой ответ, включая анализ и рекомендации, должен быть написан строго на русском языке.

ДИАЛОГ:
{dialog_text}

СОЗДАЙ РЕЗЮМЕ В ФОРМАТЕ:

1. ОСНОВНАЯ ТЕМА: [Одним предложением, о чем был разговор]

2. КЛЮЧЕВЫЕ ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:
   - [Факт 1]
   - [Факт 2]

3. ЧТО БЫЛО СДЕЛАНО/РЕШЕНО:
   - [Достижения/решения]

4. ТЕКУЩИЙ СТАТУС/НЕЗАВЕРШЕННЫЕ ВОПРОСЫ:
   - [Что еще нужно сделать/обсудить]

5. ВАЖНЫЙ КОНТЕКСТ:
   - [Любые другие важные детали: предпочтения, ограничения, договоренности]

ВАЖНО:
- Резюме должно быть фактологичным, без оценки
- Не добавляй информацию, которой нет в диалоге
- Сохрани хронологическую логику

"""  # noqa: E501
TIMEOUT = 120


md_splitter: Final[MarkdownHeaderTextSplitter] = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1")]
)

text_splitter: Final[TextSplitter] = RecursiveCharacterTextSplitter(
    chunk_size=settings.rag.chunk_size,
    chunk_overlap=settings.rag.chunk_overlap,
    length_function=len,
    separators=["\n#"],
)

TextEmbedding.add_custom_model(
    model="skatzR/USER-BGE-M3-ONNX-INT8",  # Используем имя ONNX-репозитория
    pooling=PoolingType.CLS,
    normalization=True,
    sources=ModelSource(hf="skatzR/USER-BGE-M3-ONNX-INT8"),  # Загружаем модель с Hugging Face
    dim=1024,
    model_file="model_quantized.onnx",  # Указываем имя файла с моделью
)

embeddings = TextEmbedding(model_name="skatzR/USER-BGE-M3-ONNX-INT8")


llm: ChatOpenAI = ChatOpenAI(
    api_key=SecretStr(settings.yandexcloud.api_key),
    model=f"gpt://{settings.yandexcloud.folder_id}/yandexgpt/latest",
    base_url="https://llm.api.cloud.yandex.net/v1",
    max_retries=3,
    profile=ModelProfile(max_input_tokens=32_000),
)


summarization_middleware = SummarizationMiddleware(
    model=llm,
    trigger=("fraction", 0.8),
    keep=("fraction", 0.3),
    summary_prompt=PROMPT_SUMMARIZE_CHAT,
)
