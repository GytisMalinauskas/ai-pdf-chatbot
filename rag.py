"""Core RAG (Retrieval-Augmented Generation) pipeline."""

from dataclasses import dataclass
import io
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"]


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
    """Read raw PDF bytes and return a list of (page_number, cleaned_text) pairs."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF file is empty.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except PdfReadError as exc:
        raise ValueError(f"Could not read PDF file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        # Collapse extra newlines/spaces from PDF layout into clean single spaces
        cleaned = " ".join(raw_text.split())
        if cleaned:
            pages.append((page_num, cleaned))

    if not pages:
        raise ValueError(
            "No selectable text found in this PDF. "
            "It may be a scanned image or an empty document."
        )

    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 800,
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
    client: genai.Client,
    batch_size: int = 50,
) -> np.ndarray:
    """Convert a list of strings into a 2D NumPy array of embedding vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )
        for embedding in response.embeddings:
            vectors.append(embedding.values)

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
        client: genai.Client,
        filename: str,
    ) -> None:
        """Embed all chunks and store them in memory, replacing any previous PDF."""
        texts = [c.text for c in chunks]
        self.embeddings = embed_texts(texts, client)
        self.chunks = chunks
        self.filename = filename

    def search(
        self,
        query: str,
        client: genai.Client,
        top_k: int = 4,
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most semantically similar chunks for a question."""
        if not self.is_ready or self.embeddings is None:
            raise ValueError("No document has been indexed yet.")

        query_vec = embed_texts([query], client)[0]
        scores = cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices]


def generate_answer(
    question: str,
    store: InMemoryVectorStore,
    client: genai.Client,
    top_k: int = 4,
) -> dict:
    """Retrieve relevant chunks and ask Gemini to answer strictly from them."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    results = store.search(cleaned_question, client, top_k=top_k)

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

    response = None
    last_error: Exception | None = None
    for model_name in CHAT_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            break
        except APIError as exc:
            last_error = exc
            continue

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

    return {"answer": answer_text, "sources": sources}


