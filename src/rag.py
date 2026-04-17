import contextlib
import gc
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles.tempfile
import chromadb
import markitdown
import pymupdf4llm
from aiofiles.threadpool.binary import AsyncFileIO
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .depends import embeddings, md_splitter, text_splitter
from .settings import CHROMA_PATH

INDEX_NAME = "main-index"
AVAILABLE_EXTENSIONS: tuple[str, ...] = ("doc", "docx", "pdf", "txt", "md")

logger = logging.getLogger(__name__)

client = chromadb.PersistentClient(CHROMA_PATH)
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50, length_function=len)


def batch_chunks(items: list[Any], batch_size: int = 5) -> Iterable[list[Any]]:
    """
    Асинхронный генератор батчей фиксированного размера.

    :param items: список элементов для батчинга
    :param batch_size: размер одного батча (по умолчанию 5)
    :param delay_between_batches: задержка между батчами в секундах (полезно при rate limit)
    """
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        yield batch


async def indexing(docs: list[Document], batch_size: int = 5) -> list[str]:
    """
    Индексация и добавление документа в семантический индекс с батчингом чанков.

    :param text: Текст документа.
    :param metadata: Мета-информация документа.
    :param batch_size: Количество чанков в одном батче при добавлении в векторную БД.
    :returns: Идентификаторы всех чанков в индексе.
    """

    start_time = time.monotonic()
    logger.info("Starting index document text, length %s characters", len(docs))

    collection = client.get_or_create_collection(INDEX_NAME)

    # Генерируем уникальные ID для всех чанков
    ids = [str(uuid4()) for _ in docs]

    texts = [doc.page_content for doc in docs]
    embed = embeddings.embed(texts)
    embeddings_list = [i.tolist() for i in embed]

    # Подготавливаем метаданные
    metadatas = [doc.metadata or {"source": "unknown"} for doc in docs]

    # Батчинг добавления в коллекцию
    added_ids: list[str] = []

    for batch_idx, batch_slice in enumerate(
        batch_chunks(list(range(len(docs))), batch_size=batch_size)
    ):
        batch_ids = [ids[i] for i in batch_slice]
        batch_docs = [docs[i] for i in batch_slice]
        batch_embs = [embeddings_list[i] for i in batch_slice]
        batch_metas = [metadatas[i] for i in batch_slice]

        logger.info("Adding batch %s with %s chunks to collection", batch_idx + 1, len(batch_ids))

        collection.add(
            ids=batch_ids,
            documents=[doc.page_content for doc in batch_docs],
            embeddings=batch_embs,
            metadatas=batch_metas,
        )

        added_ids.extend(batch_ids)

    logger.info(
        "Finished indexing text (%s chunks), time: %s seconds",
        len(docs),
        round(time.monotonic() - start_time, 2),
    )

    return added_ids


def clean_text(text: str) -> str:
    """Очистка текста от экранированных символов и Unicode"""
    if not isinstance(text, str):
        return str(text)

    # Метод 1: Декодирование Unicode escape последовательностей
    with contextlib.suppress(UnicodeDecodeError):
        text = text.encode("utf-8").decode("unicode_escape")

    # Метод 2: Обработка JSON строк
    try:
        # Убираем лишние кавычки в начале и конце если есть
        if text.startswith('"') and text.endswith('"'):
            text = json.loads(text)
        else:
            text = json.loads(f'"{text}"')
    except (json.JSONDecodeError, TypeError):
        pass

    # Метод 3: Ручная замена Unicode последовательностей
    def replace_unicode(match):
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    return re.sub(r"\\u([0-9a-fA-F]{4})", replace_unicode, text)


async def retrieve(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    search_string: str | None = None,
    n_results: int = 10,
) -> list[str]:
    """Извлечение документов с очисткой текста"""

    collection = client.get_collection(INDEX_NAME)
    logger.info("Retrieving for query: '%s...'", query[:50])

    embed = embeddings.query_embed(query)
    embeddings_list = [i.tolist() for i in embed]
    embed = embeddings.query_embed(query)   # ожидается [[...]]
    params = {"query_embeddings": embeddings_list, "n_results": n_results}

    if metadata_filter:
        if len(metadata_filter) == 0:
            pass  # пустой фильтр не передаём
        elif len(metadata_filter) == 1:
            params["where"] = metadata_filter
        else:
            # Несколько полей → оборачиваем в $and
            params["where"] = {"$and": [{k: v} for k, v in metadata_filter.items()]}
    if search_string:
        params["where_document"] = {"$contains": search_string}

    result = collection.query(**params, include=["documents", "metadatas", "distances"])

    cleaned_results = []
    for document, metadata, distance in zip(
        result["documents"][0],  # type: ignore  # noqa: PGH003
        result["metadatas"][0],  # type: ignore  # noqa: PGH003
        result["distances"][0],  # type: ignore  # noqa: PGH003
        strict=False,  # type: ignore  # noqa: PGH003
    ):
        # Очищаем документ
        cleaned_doc = clean_text(document)

        # Очищаем метаданные если нужно
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                cleaned_metadata[key] = clean_text(value)
            else:
                cleaned_metadata[key] = value

        cleaned_results.append(
            f"**Relevance score:** {round(distance, 2)}\n"
            f"**Source:** {cleaned_metadata.get('source', '')}\n"
            f"**Category:** {cleaned_metadata.get('category', '')}\n"
            "**Document:**\n"
            f"{cleaned_doc}"
        )

    return cleaned_results


@asynccontextmanager
async def open_temp_file(data: bytes, suffix: str) -> AsyncGenerator[AsyncFileIO]:
    """Контекстный менеджер для открытия и работы с временным файлом.

    :param data: Данные (поток байтов), которые нужно записать в файл.
    :param suffix: Суффикс временного файла.
    """
    async with aiofiles.tempfile.NamedTemporaryFile(mode="wb", suffix=suffix) as file:
        await file.write(data)
        await file.flush()
        yield file  # type: ignore  # noqa: PGH003


async def process_file(file: Any) -> list[Document]:  # или более точный тип, если есть
    """Обрабатывает файл и возвращает список Document."""
    # === Исправление 1: надёжное получение имени файла ===
    file_path = Path(file.name)  # Path всегда имеет .rsplit и .suffix
    filename_str = str(file_path)
    extension = file_path.suffix.lower().lstrip(".")

    try:
        match extension:
            case "docx" | "doc":
                # === Исправление 2: правильное использование MarkItDown ===
                md = markitdown.MarkItDown()
                # convert принимает путь (str или Path) — мы передаём строку
                result = md.convert(filename_str)
                md_text = result.text_content  # или result.markdown в старых версиях

            case "pdf":
                md_text = pymupdf4llm.to_markdown(filename_str)

                gc.collect()

            case _:
                # Для текстовых файлов читаем содержимое
                content = await file.read()
                if isinstance(content, bytes):
                    md_text = content.decode("utf-8", errors="replace")
                else:
                    md_text = str(content)

        logger.info("File %s successfully processed", filename_str)
        return md_splitter.split_text(md_text)  # type: ignore  # noqa: PGH003

    except Exception as exc:
        logger.exception("Failed to process file %s: %s", filename_str, exc)  # noqa: TRY401
        # Fallback
        fallback_text = "Не удалось извлечь содержимое документа."
        return md_splitter.split_text(fallback_text)  # type: ignore  # noqa: PGH003


async def indexing_file(data: bytes, filename: str) -> list[Document]:
    async with open_temp_file(
        data, suffix=f".{filename.rsplit('.', maxsplit=1)[-1]}"
    ) as temp_file:
        documents = await process_file(temp_file)
        documents = text_splitter.split_documents(documents)
        await indexing(docs=documents)
        return documents
