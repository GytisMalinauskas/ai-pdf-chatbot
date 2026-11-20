# AI PDF Chatbot (RAG from Scratch)

A fast, lightweight **Retrieval-Augmented Generation (RAG)** web application that lets you upload any PDF and ask questions grounded strictly in its contents—complete with page citations, Markdown rendering, and per-stage latency metrics.

Built from scratch in pure Python (without heavy orchestration frameworks) to demonstrate how PDF parsing, chunking, local embeddings, vector similarity search, and LLM prompting work under the hood.

---

## Features

- **Fast PDF Extraction:** Uses **PyMuPDF (`fitz`)** to parse hundreds of pages in ~1–2 seconds.
- **Local Embeddings (No Rate Limits):** Generates embeddings locally via **FastEmbed** (`BAAI/bge-small-en-v1.5`) with automatic **CUDA GPU** acceleration and CPU fallback.
- **Pure NumPy Vector Search:** Computes cosine similarity in-memory in < 1 ms.
- **Grounded Gemini Answers:** Uses the **Google Gemini API** (`gemini-3.5-flash-lite` with automatic model fallbacks) and instructs the model to answer only from retrieved excerpts.
- **Modern Web UI:** Clean **Tailwind CSS** dark interface with drag-and-drop PDF upload, sanitized Markdown rendering (`marked` + `DOMPurify`), page citation chips, and live timing breakdowns.
- **Automated Tests:** Offline unit and API test suite using `unittest` and mocked clients.

---

## Architecture

```text
Browser (Tailwind CSS + Vanilla JS)
   │
   ├── POST /upload (PDF) ──▶ FastAPI ──▶ PyMuPDF ──▶ Overlapping Chunks ──▶ FastEmbed (Local) ──▶ In-Memory NumPy Store
   │
   └── POST /ask (Question) ─▶ FastAPI ──▶ FastEmbed Query Vector ──▶ Cosine Similarity (Top-K) ──▶ Gemini API ──▶ Grounded Answer + Citations
```

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.11+, FastAPI, Uvicorn | REST API, file upload validation, static UI serving |
| **PDF Parsing** | PyMuPDF (`fitz`) | High-speed text extraction with page numbers |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`) | Local ONNX embeddings (CPU or NVIDIA CUDA) |
| **Vector Search** | NumPy | Cosine similarity ranking |
| **LLM** | Google GenAI SDK (`google-genai`) | Grounded answer generation |
| **Frontend** | HTML, Tailwind CSS, Marked, DOMPurify | Responsive chat UI & safe Markdown rendering |

---

## Project Structure

```text
ai-pdf-chatbot/
├── app.py               # FastAPI server & HTTP endpoints (/upload, /ask, /health)
├── rag.py               # Core RAG pipeline (PDF parsing, chunking, embeddings, search, Gemini)
├── static/
│   └── index.html       # Tailwind CSS chat interface
├── test_rag.py          # Unit tests for RAG logic
├── test_app.py          # FastAPI endpoint tests
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/GytisMalinauskas/ai-pdf-chatbot.git
cd ai-pdf-chatbot
```

### 2. Create a virtual environment & install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

*(Optional — for NVIDIA CUDA GPU embeddings, install CUDA Toolkit 12.x + cuDNN 9.x and swap `fastembed` for `fastembed-gpu`.)*

### 3. Configure your API key
Create a `.env` file in the project root and add your key from [Google AI Studio](https://aistudio.google.com/apikey):
```env
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the application
```powershell
.\.venv\Scripts\uvicorn.exe app:app --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser (or **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** for interactive API docs).

---

## Running Tests

All unit and API tests run offline using mocks (no API quota used):
```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

---

## Roadmap

- Disk caching (`SHA-256`) for instant re-loading of previously indexed PDFs
- Streaming LLM responses (SSE)
- Multi-turn conversation memory
- Multi-PDF library with persistent vector storage (ChromaDB / Qdrant)
- Hybrid search (Vector + BM25) & cross-encoder reranking
