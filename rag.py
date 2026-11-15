"""Core RAG (Retrieval-Augmented Generation) pipeline."""

from dataclasses import dataclass
import io
from pypdf import PdfReader
from pypdf.errors import PdfReadError


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
