"""API tests for app.py using FastAPI's TestClient and a mocked Gemini client."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np

import app as app_module
from rag import get_embedding_model, get_genai_client


def build_minimal_pdf(text: str) -> bytes:
    """Create a valid single-page PDF containing the given ASCII text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        ),
        (
            f"4 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
            + stream
            + b"\nendstream endobj\n"
        ),
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(pdf)


class TestApiEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        # Reset in-memory vector store before every test for isolation
        app_module.vector_store.filename = None
        app_module.vector_store.chunks = []
        app_module.vector_store.embeddings = None

        self.mock_embedder = MagicMock()
        self.mock_embedder.embed.return_value = [np.array([1.0, 0.0], dtype=np.float32)]

        self.mock_client = MagicMock()
        self.mock_client.models.generate_content.return_value = SimpleNamespace(
            text="FastAPI is used to build Python APIs (Page 1)."
        )

        app_module.app.dependency_overrides[get_embedding_model] = lambda: self.mock_embedder
        app_module.app.dependency_overrides[get_genai_client] = lambda: self.mock_client
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        app_module.app.dependency_overrides.clear()

    def test_home_serves_html_ui(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("AI PDF Chatbot", res.text)

    def test_health_initially_has_no_document(self) -> None:
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["document_loaded"])

    def test_ask_before_upload_returns_400(self) -> None:
        res = self.client.post("/ask", json={"question": "What is this about?"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("upload a PDF", res.json()["detail"])

    def test_upload_non_pdf_returns_400(self) -> None:
        res = self.client.post(
            "/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn(".pdf", res.json()["detail"])

    def test_upload_corrupted_pdf_returns_400(self) -> None:
        res = self.client.post(
            "/upload",
            files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
        )
        self.assertEqual(res.status_code, 400)

    def test_upload_and_ask_end_to_end(self) -> None:
        pdf_bytes = build_minimal_pdf("FastAPI is used to build Python APIs.")
        upload_res = self.client.post(
            "/upload",
            files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(upload_res.status_code, 200)
        self.assertEqual(upload_res.json()["filename"], "guide.pdf")
        self.assertEqual(upload_res.json()["pages"], 1)
        self.assertGreaterEqual(upload_res.json()["chunks"], 1)
        self.assertIn("embed_ms", upload_res.json()["timings"])

        ask_res = self.client.post("/ask", json={"question": "What is FastAPI used for?"})
        self.assertEqual(ask_res.status_code, 200)
        data = ask_res.json()
        self.assertEqual(data["answer"], "FastAPI is used to build Python APIs (Page 1).")
        self.assertEqual(data["sources"][0]["page"], 1)
        self.assertIn("llm_generate_ms", data["timings"])


if __name__ == "__main__":
    unittest.main()
