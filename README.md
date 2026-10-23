# Botzi AI Service

High-speed Ingestion, Hybrid Retrieval (Dense + BM25), Cross-Encoder Reranking, and Grounded Generation Service built for the Botzi AI Website Bot SaaS platform according to `ai_bot_saas_system_architecture.md`.

---

## Architecture Implementation

1. **Structure-Aware Ingestion (`app/ingestion/`)**:
   - Parsers for Markdown, Plain Text, PDF (`pypdf`), Word (`python-docx`), and Excel spreadsheets (`openpyxl`).
   - Semantic chunker (`app/ingestion/chunker.py`) with configurable token limits (~300–700 tokens), paragraph/sentence boundary splitting, overlap preservation, and metadata enrichment (`organization_id`, `bot_id`, `document_id`, `page_number`, `section_title`).

2. **Embeddings (`app/embeddings/`)**:
   - **Gemini Embeddings:** Google GenAI SDK integration (`text-embedding-004`) with batching for high throughput.
   - **Local Dense Fallback:** FastEmbed ONNX runtime (`BAAI/bge-small-en-v1.5` / `bge-base-en-v1.5`).
   - **BM25 Sparse Embeddings:** FastEmbed BM25 (`Qdrant/bm25`) for exact keyword, invoice, and SKU lookup.

3. **Multi-Tenant Vector Store (`app/vector_store/`)**:
   - Qdrant integration with named vectors (`dense` and `sparse`).
   - Strict payload-based multi-tenancy (`organization_id` & `bot_id` filters applied to all queries).
   - In-memory support for testing and production remote cluster support.

4. **Hybrid Retrieval & Reranker (`app/retrieval/`)**:
   - Hybrid search fusing Dense Semantic Search and BM25 Sparse Search via Reciprocal Rank Fusion (RRF).
   - Fast Cross-Encoder Reranker (`Xenova/ms-marco-MiniLM-L-6-v2`) with sigmoid-normalized relevance scores.

5. **Grounded Generation & Hallucination Defense (`app/generation/`)**:
   - **Answerability Classifier:** Evaluates evidence confidence and triggers safe abstention when knowledge is insufficient.
   - **Prompt Builder:** Strictly bounds evidence within `<EVIDENCE>` blocks, formats source citations (`[Source 1]`), and defends against indirect prompt injection.
   - **Gemini LLM Provider:** Google GenAI SDK integration supporting factual grounding (`gemini-2.5-flash`) and token-by-token streaming.

6. **FastAPI Endpoints (`app/api/`)**:
   - `POST /api/v1/ingestion/process`: Ingest and index raw text, PDF, DOCX, or XLSX.
   - `DELETE /api/v1/ingestion/documents/{document_id}`: Remove indexed document chunks.
   - `POST /api/v1/search/hybrid`: Direct hybrid search & scoring test endpoint.
   - `POST /api/v1/chat/completions`: Grounded chat completion with Server-Sent Events (SSE) streaming (`event: citation`, `event: token`, `event: done`).
   - `GET /health`: Health and subsystem status.

---

## Quickstart

### 1. Requirements
- Python 3.12+
- `uv` package manager

### 2. Install Dependencies
```bash
cd ai-service
uv sync
```

### 3. Run Tests
```bash
uv run pytest
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env and insert your GEMINI_API_KEY if testing against live Google GenAI
```

### 5. Start Service
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://localhost:8000/docs`
