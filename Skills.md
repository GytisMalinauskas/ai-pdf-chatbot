---
name: pdf-rag-dev
description: Guidelines for developing and testing the local Python PDF RAG chatbot. Use when editing RAG pipeline code, FastAPI endpoints, or running project tests.
---

# PDF RAG Chatbot Skill (Example)

> **Note:** In Antigravity, skills are named `SKILL.md` inside `.agent/skills/<skill-name>/SKILL.md` so the agent discovers them automatically. This file is a reference example in the project root.

## Project Conventions
- Always run Python commands using the local virtual environment: `.\.venv\Scripts\python.exe`.
- Never hardcode or commit `GEMINI_API_KEY`; always load it from `.env` via `python-dotenv`.
- Keep RAG logic in `rag.py` and HTTP routing/validation in `app.py`.

## Verification Checklist
1. Check `extract_pages_from_pdf` handles empty or corrupted PDFs with a `ValueError`.
2. Verify `InMemoryVectorStore.search` returns chunks sorted by descending cosine similarity.
3. Run the end-to-end test after any backend change.
