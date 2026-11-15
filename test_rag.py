"""Unit tests for rag.py (runs offline using mock Gemini responses)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from rag import (
    Chunk,
    InMemoryVectorStore,
    chunk_pages,
    cosine_similarity,
    extract_pages_from_pdf,
    generate_answer,
)


def make_mock_client(
    embeddings: list[list[float]] | None = None,
    answer_text: str = "Mocked answer (Page 1).",
) -> MagicMock:
    """Create a fake Gemini client so tests run fast and offline."""
    client = MagicMock()

    if embeddings is not None:
        client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=vec) for vec in embeddings]
        )

    client.models.generate_content.return_value = SimpleNamespace(text=answer_text)
    return client


class TestPdfAndChunking(unittest.TestCase):
    def test_empty_pdf_bytes_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_pages_from_pdf(b"")

    def test_invalid_pdf_bytes_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_pages_from_pdf(b"not a real pdf")

    def test_chunk_pages_creates_overlapping_chunks(self) -> None:
        pages = [(1, "A" * 1200)]
        chunks = chunk_pages(pages, chunk_size=800, overlap=150)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].text), 800)
        self.assertEqual(len(chunks[1].text), 550)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[1].chunk_index, 1)

    def test_chunk_pages_invalid_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            chunk_pages([(1, "hello")], chunk_size=100, overlap=100)


class TestVectorSearch(unittest.TestCase):
    def test_cosine_similarity_ranks_matching_vector_highest(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        docs = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32)

        scores = cosine_similarity(query, docs)
        self.assertAlmostEqual(float(scores[0]), 0.0, places=5)
        self.assertAlmostEqual(float(scores[1]), 1.0, places=5)
        self.assertEqual(int(np.argmax(scores)), 1)

    def test_vector_store_search_returns_top_k_sorted(self) -> None:
        chunks = [
            Chunk(text="Cats sleep all day.", page=1, chunk_index=0),
            Chunk(text="Rockets launch into orbit.", page=2, chunk_index=1),
        ]
        index_client = make_mock_client(embeddings=[[1.0, 0.0], [0.0, 1.0]])

        store = InMemoryVectorStore()
        store.index_chunks(chunks, index_client, filename="sample.pdf")

        search_client = make_mock_client(embeddings=[[0.0, 1.0]])
        results = store.search("space launch", search_client, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page, 2)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_search_before_indexing_raises(self) -> None:
        store = InMemoryVectorStore()
        with self.assertRaises(ValueError):
            store.search("anything", make_mock_client(), top_k=1)


class TestGenerateAnswer(unittest.TestCase):
    def test_empty_question_raises(self) -> None:
        store = InMemoryVectorStore()
        with self.assertRaises(ValueError):
            generate_answer("   ", store, make_mock_client())

    def test_generate_answer_returns_answer_and_sources(self) -> None:
        chunks = [Chunk(text="FastAPI is fast.", page=1, chunk_index=0)]
        store = InMemoryVectorStore()
        store.index_chunks(chunks, make_mock_client(embeddings=[[1.0, 0.0]]), "doc.pdf")

        ask_client = make_mock_client(
            embeddings=[[1.0, 0.0]],
            answer_text="FastAPI is fast (Page 1).",
        )
        result = generate_answer("What is FastAPI?", store, ask_client, top_k=1)

        self.assertEqual(result["answer"], "FastAPI is fast (Page 1).")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["page"], 1)


if __name__ == "__main__":
    unittest.main()
