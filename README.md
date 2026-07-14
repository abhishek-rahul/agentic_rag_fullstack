# Full-Stack Agentic RAG GenAI Application

A complete local-first full-stack GenAI project with:

- Python FastAPI backend
- React + Vite frontend
- LangChain RAG pipeline
- LangGraph workflow
- FAISS local vector store
- SQLite persistent conversation memory
- Short-term in-process session memory
- OpenAI chat model support
- Ollama local chat model support

This project includes optional request and RAG-result guardrails controlled by
`GUARDRAIL_ENABLED`. It does **not** include evals, scoring, benchmarking,
LLM-as-judge, RAG evaluation, agent evaluation, promptfoo, DeepEval, Ragas,
TruLens, LangSmith evals, or OpenAI evals.

## Project structure

```text
agentic_rag_fullstack/
  backend/
    app/
      main.py
      api/
        chat_routes.py
      core/
        config.py
      domain/
        models.py
      services/
        llm_gateway.py
        rag_service.py
        memory_service.py
        chat_service.py
        graph_service.py
      infrastructure/
        vector_store.py
        sqlite_store.py
      data/
        docs/
          sample_company_policy.txt
        faiss_index/
        memory.db
    requirements.txt
    .env.example
    README.md
  frontend/
    src/
      components/
        ChatBox.jsx
        MessageBubble.jsx
      api/
        chatApi.js
      App.jsx
      main.jsx
      styles.css
    package.json
    vite.config.js
    README.md
  .gitignore
  README.md
```

## Architecture overview

### Backend flow

```text
React frontend
   ↓ POST /chat
FastAPI route
   ↓
ChatService
   ↓
LangGraph workflow:
   1. load_memory
   2. retrieve_context
   3. generate_answer
   4. save_memory
   ↓
Response with answer + sources
```

### Memory design

```text
Short-term memory:
  In-process recent turns per session_id

Long-term memory:
  SQLite table storing session_id, role, content, timestamp
```

### RAG design

```text
backend/app/data/docs/*.txt
   ↓
LangChain document loader
   ↓
RecursiveCharacterTextSplitter
   ↓
Embeddings via Ollama or OpenAI
   ↓
FAISS local index
   ↓
Retriever used during /chat
```

### LLM gateway

The backend uses `app/services/llm_gateway.py` to switch between:

```text
provider = openai
provider = ollama
```

No API keys are hardcoded. OpenAI credentials are read from `backend/.env`.

## Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

## Ollama setup

Install Ollama, then run:

```bash
ollama serve
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
```

The project uses:

```text
OLLAMA_DEFAULT_MODEL=qwen2.5:0.5b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDINGS_PROVIDER=ollama
```

Then ingest docs:

```bash
curl -X POST http://localhost:8000/ingest
```

Chat with Ollama:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the refund policy?","session_id":"demo-session","provider":"ollama","model":"qwen2.5:0.5b"}'
```

## OpenAI setup

Edit `backend/.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_DEFAULT_MODEL=gpt-4o-mini
```

For OpenAI embeddings, also set:

```bash
EMBEDDINGS_PROVIDER=openai
```

Then ingest docs again because the FAISS index must be built with the same embedding provider used at query time:

```bash
curl -X POST http://localhost:8000/ingest
```

Chat with OpenAI:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the refund policy?","session_id":"demo-session","provider":"openai","model":"gpt-4o-mini"}'
```

## Frontend setup

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Then:

1. Click **Ingest Docs**.
2. Select provider: `ollama` or `openai`.
3. Enter model name.
4. Ask: `What is the refund policy?`

## API contracts

### GET /health

Response:

```json
{
  "status": "ok",
  "app_name": "Local Agentic RAG Assistant",
  "environment": "local"
}
```

### POST /ingest

Ingests local documents from:

```text
backend/app/data/docs
```

Response:

```json
{
  "status": "completed",
  "documents_loaded": 1,
  "chunks_created": 1,
  "vector_store_path": ".../backend/app/data/faiss_index"
}
```

### POST /chat

Request:

```json
{
  "message": "What is the refund policy?",
  "session_id": "unique-session-id",
  "provider": "openai or ollama",
  "model": "model name"
}
```

Response:

```json
{
  "answer": "assistant answer",
  "session_id": "same-session-id",
  "provider": "selected provider",
  "model": "selected model",
  "sources": []
}
```

## Notes

- Run `/ingest` before `/chat` if you want document-based answers.
- SQLite database is created automatically at `backend/app/data/memory.db`.
- FAISS index is saved locally at `backend/app/data/faiss_index`.
- If you switch embedding provider, re-run `/ingest`.
- If Ollama is not running, `/chat` or `/ingest` with Ollama embeddings will fail with a connection error.
