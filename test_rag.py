"""Unit tests for rag.py (runs offline using mock embedder and mock Gemini)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from rag import (
    Chunk,
    InMemoryVectorStore,
    chunk_pages,
    cosine_similarity,
    embed_texts,
    extract_pages_from_pdf,
    generate_answer,
)


def make_mock_embedder(vectors: list[list[float]]) -> MagicMock:
    """Create a fake local TextEmbedding model returning fixed vectors."""
    embedder = MagicMock()
    embedder.embed.return_value = [np.array(vec, dtype=np.float32) for vec in vectors]
    return embedder


def make_mock_client(answer_text: str = "Mocked answer (Page 1).") -> MagicMock:
    """Create a fake Gemini client for chat generation."""
    client = MagicMock()
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
        index_embedder = make_mock_embedder([[1.0, 0.0], [0.0, 1.0]])

        store = InMemoryVectorStore()
        store.index_chunks(chunks, filename="sample.pdf", embedder=index_embedder)

        search_embedder = make_mock_embedder([[0.0, 1.0]])
        results = store.search("space launch", top_k=1, embedder=search_embedder)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page, 2)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_search_before_indexing_raises(self) -> None:
        store = InMemoryVectorStore()
        with self.assertRaises(ValueError):
            store.search("anything", top_k=1, embedder=make_mock_embedder([[1.0, 0.0]]))

    @patch("rag.get_embedding_model")
    def test_embed_texts_falls_back_to_cpu_on_cuda_error(
        self, mock_get_model: MagicMock
    ) -> None:
        gpu_embedder = MagicMock()
        gpu_embedder.embed.side_effect = RuntimeError("cuDNN is unavailable: cudnn64_9.dll")
        cpu_embedder = make_mock_embedder([[0.25, 0.75]])
        mock_get_model.side_effect = [gpu_embedder, cpu_embedder]

        result = embed_texts(["hello"])
        self.assertEqual(result.shape, (1, 2))
        mock_get_model.assert_any_call(force_cpu=True)


class TestGenerateAnswer(unittest.TestCase):
    def test_empty_question_raises(self) -> None:
        store = InMemoryVectorStore()
        with self.assertRaises(ValueError):
            generate_answer("   ", store, make_mock_client())

    def test_generate_answer_returns_answer_and_sources(self) -> None:
        chunks = [Chunk(text="FastAPI is fast.", page=1, chunk_index=0)]
        store = InMemoryVectorStore()
        store.index_chunks(chunks, "doc.pdf", embedder=make_mock_embedder([[1.0, 0.0]]))

        ask_embedder = make_mock_embedder([[1.0, 0.0]])
        ask_client = make_mock_client(answer_text="FastAPI is fast (Page 1).")
        result = generate_answer(
            "What is FastAPI?", store, ask_client, top_k=1, embedder=ask_embedder
        )

        self.assertEqual(result["answer"], "FastAPI is fast (Page 1).")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["page"], 1)
        self.assertIn("llm_generate_ms", result["timings"])
        self.assertIn("vector_search_ms", result["timings"])


if __name__ == "__main__":
    unittest.main()
