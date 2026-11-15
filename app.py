"""FastAPI backend for the PDF RAG Chatbot."""

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel

from rag import (
    InMemoryVectorStore,
    chunk_pages,
    extract_pages_from_pdf,
    generate_answer,
    get_genai_client,
)

MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB upload limit
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AI PDF Chatbot")
vector_store = InMemoryVectorStore()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def serve_home():
    """Serve the web interface if present, otherwise return API status."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "ok", "message": "Backend is running. Add static/index.html for UI."}


@app.get("/health")
def health_check() -> dict:
    """Return server and indexed-document status."""
    return {
        "status": "ok",
        "document_loaded": vector_store.is_ready,
        "filename": vector_store.filename,
        "chunks": len(vector_store.chunks),
    }


@app.post("/upload")
async def upload_pdf(
    file: Annotated[UploadFile, File(...)],
    client: Annotated[genai.Client, Depends(get_genai_client)],
) -> dict:
    """Extract text from an uploaded PDF, chunk it, and index its embeddings."""
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDF exceeds the 20 MB size limit.")

    try:
        pages = extract_pages_from_pdf(pdf_bytes)
        chunks = chunk_pages(pages)
        vector_store.index_chunks(chunks, client, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to index PDF: {exc}") from exc

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "message": "PDF uploaded and indexed successfully.",
    }


@app.post("/ask")
def ask_question(
    payload: AskRequest,
    client: Annotated[genai.Client, Depends(get_genai_client)],
) -> dict:
    """Answer a user's question using the currently indexed PDF."""
    if not vector_store.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Please upload a PDF first.",
        )

    try:
        return generate_answer(payload.question, vector_store, client)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to generate answer: {exc}") from exc
