"""FastAPI backend for the PDF RAG Chatbot."""

from pathlib import Path
import time
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from google import genai
from pydantic import BaseModel

from rag import (
    InMemoryVectorStore,
    chunk_pages,
    extract_pages_from_pdf,
    generate_answer,
    get_embedding_model,
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
    embedder: Annotated[TextEmbedding, Depends(get_embedding_model)],
) -> dict:
    """Extract text from an uploaded PDF, chunk it, and index its embeddings locally."""
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDF exceeds the 20 MB size limit.")

    try:
        t0 = time.perf_counter()
        pages = extract_pages_from_pdf(pdf_bytes)
        t1 = time.perf_counter()
        chunks = chunk_pages(pages)
        t2 = time.perf_counter()
        vector_store.index_chunks(chunks, filename=filename, embedder=embedder)
        t3 = time.perf_counter()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc

    timings = {
        "extract_ms": round((t1 - t0) * 1000, 1),
        "chunk_ms": round((t2 - t1) * 1000, 1),
        "embed_ms": round((t3 - t2) * 1000, 1),
        "total_ms": round((t3 - t0) * 1000, 1),
    }
    print(f"[UPLOAD TIMINGS] {filename}: {timings}")

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "timings": timings,
        "message": "PDF uploaded and indexed successfully.",
    }


@app.post("/ask")
def ask_question(
    payload: AskRequest,
    client: Annotated[genai.Client, Depends(get_genai_client)],
    embedder: Annotated[TextEmbedding, Depends(get_embedding_model)],
) -> dict:
    """Answer a user's question using the currently indexed PDF."""
    if not vector_store.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Please upload a PDF first.",
        )

    try:
        result = generate_answer(payload.question, vector_store, client, embedder=embedder)
        print(f"[ASK TIMINGS] {result['timings']}")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to generate answer: {exc}") from exc
