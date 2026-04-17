from fastapi import APIRouter, File, UploadFile, status
from langchain_core.documents import Document

from ..rag import indexing_file

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    path="/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=list[Document],
    summary="Загружает документы в базу знаний",
)
async def upload_documents(file: UploadFile = File(...)) -> list[Document]:
    filedata = await file.read()
    return await indexing_file(filedata, file.filename)  # type: ignore  # noqa: PGH003
