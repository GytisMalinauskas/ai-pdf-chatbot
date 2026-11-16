"""Core RAG (Retrieval-Augmented Generation) pipeline with local embeddings."""

from dataclasses import dataclass
import io
import os
import time
from dotenv import load_dotenv
from fastembed import TextEmbedding
import fitz  # PyMuPDF
from google import genai
from google.genai.errors import APIError
import numpy as np
import onnxruntime as ort

LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHAT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]

_embedding_model: TextEmbedding | None = None


def get_embedding_model(force_cpu: bool = False) -> TextEmbedding:
    """Lazily load and cache the local ONNX embedding model (uses CUDA if available)."""
    global _embedding_model
    if _embedding_model is None or force_cpu:
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if not force_cpu and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        _embedding_model = TextEmbedding(
            model_name=LOCAL_EMBEDDING_MODEL,
            providers=providers,
        )
    return _embedding_model


def get_genai_client() -> genai.Client:
    """Load the API key from .env and return an authenticated Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your key to the .env file."
        )
    return genai.Client(api_key=api_key)


@dataclass
class Chunk:
    """A single piece of text extracted from a PDF, with its source metadata."""

    text: str
    page: int
    chunk_index: int


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Read raw PDF bytes with PyMuPDF and return (page_number, cleaned_text) pairs."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF file is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {exc}") from exc

    pages: list[tuple[int, str]] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            cleaned = " ".join(raw_text.split())
            if cleaned:
                pages.append((page_num, cleaned))
    finally:
        doc.close()

    if not pages:
        raise ValueError(
            "No selectable text found in this PDF. "
            "It may be a scanned image or an empty document."
        )

    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Split extracted pages into overlapping text chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    chunk_index = 0

    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, chunk_index=chunk_index))
                chunk_index += 1
            if end >= len(text):
                break
            start += step

    return chunks


def embed_texts(
    texts: list[str],
    embedder: TextEmbedding | None = None,
) -> np.ndarray:
    """Convert a list of strings into a 2D NumPy array using local embeddings."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    model = embedder or get_embedding_model()
    try:
        vectors = list(model.embed(texts))
    except Exception as exc:
        # If CUDA/cuDNN fails at runtime (e.g., missing cudnn64_9.dll), fall back to CPU
        is_cuda_err = any(k in str(exc).lower() for k in ("cuda", "cudnn"))
        if is_cuda_err and (embedder is None or embedder is _embedding_model):
            model = get_embedding_model(force_cpu=True)
            vectors = list(model.embed(texts))
        else:
            raise

    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and all document vectors."""
    query_norm = np.linalg.norm(query_vec) + 1e-10
    doc_norms = np.linalg.norm(doc_matrix, axis=1) + 1e-10
    return (doc_matrix @ query_vec) / (doc_norms * query_norm)


class InMemoryVectorStore:
    """Simple in-memory vector store for the currently uploaded PDF."""

    def __init__(self) -> None:
        self.filename: str | None = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    @property
    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    def index_chunks(
        self,
        chunks: list[Chunk],
        filename: str,
        embedder: TextEmbedding | None = None,
    ) -> None:
        """Embed all chunks locally and store them in memory."""
        texts = [c.text for c in chunks]
        self.embeddings = embed_texts(texts, embedder=embedder)
        self.chunks = chunks
        self.filename = filename

    def search(
        self,
        query: str,
        top_k: int = 4,
        embedder: TextEmbedding | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most semantically similar chunks for a question."""
        results, _ = self.search_with_timings(query, top_k=top_k, embedder=embedder)
        return results

    def search_with_timings(
        self,
        query: str,
        top_k: int = 4,
        embedder: TextEmbedding | None = None,
    ) -> tuple[list[tuple[Chunk, float]], dict[str, float]]:
        """Return (results, timings_ms) for query embedding and cosine similarity."""
        if not self.is_ready or self.embeddings is None:
            raise ValueError("No document has been indexed yet.")

        t0 = time.perf_counter()
        query_vec = embed_texts([query], embedder=embedder)[0]
        t1 = time.perf_counter()
        scores = cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        t2 = time.perf_counter()

        results = [(self.chunks[idx], float(scores[idx])) for idx in top_indices]
        timings = {
            "embed_query_ms": round((t1 - t0) * 1000, 1),
            "vector_search_ms": round((t2 - t1) * 1000, 1),
        }
        return results, timings


def generate_answer(
    question: str,
    store: InMemoryVectorStore,
    client: genai.Client,
    top_k: int = 4,
    embedder: TextEmbedding | None = None,
) -> dict:
    """Retrieve relevant chunks locally and ask Gemini to answer strictly from them."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    t_start = time.perf_counter()
    results, search_timings = store.search_with_timings(
        cleaned_question, top_k=top_k, embedder=embedder
    )

    context_blocks = [
        f"[Excerpt {i} | Page {chunk.page}]\n{chunk.text}"
        for i, (chunk, _score) in enumerate(results, start=1)
    ]
    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant answering questions about an uploaded PDF.\n"
        "Use ONLY the document excerpts below to answer the question.\n"
        "If the answer is not in the excerpts, say: "
        '"I couldn\'t find that information in the uploaded document."\n'
        "Mention the page number(s) when citing facts.\n\n"
        f"--- DOCUMENT EXCERPTS ---\n{context_text}\n\n"
        f"--- QUESTION ---\n{cleaned_question}\n"
    )

    t_llm_start = time.perf_counter()
    response = None
    used_model = ""
    last_error: Exception | None = None
    for model_name in CHAT_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            used_model = model_name
            break
        except APIError as exc:
            last_error = exc
            continue
    t_llm_end = time.perf_counter()

    if response is None:
        raise RuntimeError(f"All Gemini chat models failed: {last_error}") from last_error

    answer_text = (response.text or "").strip() or "No response generated."
    sources = [
        {
            "page": chunk.page,
            "score": round(score, 4),
            "excerpt": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
        }
        for chunk, score in results
    ]

    timings = {
        **search_timings,
        "llm_generate_ms": round((t_llm_end - t_llm_start) * 1000, 1),
        "total_ms": round((t_llm_end - t_start) * 1000, 1),
    }
    return {"answer": answer_text, "model": used_model, "sources": sources, "timings": timings}
