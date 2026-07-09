# Backend - Local Agentic RAG Assistant

FastAPI backend with LangChain, LangGraph, FAISS, SQLite memory, OpenAI support, and Ollama support.

This backend intentionally contains no guardrails, evals, scoring, test harness, benchmarking, or safety evaluation layer. It focuses only on the core GenAI workflow.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
pip install -r requirements.txt
cp .env.example .env
```

## Ollama local setup

In a separate terminal:

```bash
ollama serve
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
```

The project uses `nomic-embed-text:latest` for local embeddings by default.

## OpenAI setup

Edit `backend/.env`:

```bash
OPENAI_API_KEY=your_api_key_here
EMBEDDINGS_PROVIDER=openai   # optional, only if you want OpenAI embeddings
```

## Run server

```bash
uvicorn app.main:app --reload --port 8000
```

## API endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Ingest documents into FAISS

```bash
curl -X POST http://localhost:8000/ingest
```

### Chat

Ollama:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the refund policy?","session_id":"demo-session","provider":"ollama","model":"qwen2.5:0.5b"}'
```

OpenAI:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the refund policy?","session_id":"demo-session","provider":"openai","model":"gpt-4o-mini"}'
```

## Backend architecture

```text
app/
  main.py
  api/chat_routes.py
  core/config.py
  domain/models.py
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
    faiss_index/
    memory.db
```

## Workflow

```text
/chat request
   -> LangGraph load_memory
   -> LangGraph retrieve_context
   -> LangGraph generate_answer
   -> LangGraph save_memory
   -> response with answer and sources
```
