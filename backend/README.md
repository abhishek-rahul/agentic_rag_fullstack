# Backend - Local Agentic RAG Assistant

FastAPI backend with LangChain, LangGraph, FAISS, SQLite memory, OpenAI support, and Ollama support.

This backend includes optional request and RAG-result guardrails plus a small
local regression test suite. It does not include an external evaluation
framework, scoring system, or benchmarking layer.

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

## Guardrail feature flag

Guardrails are enabled by default. To bypass both request and RAG-result
guardrail checks, set this in `backend/.env` and restart the backend:

```bash
GUARDRAIL_ENABLED=false
```

## Local Guardrail and Regression Tests

The reusable regression dataset is stored at:

```text
app/data/eval_dataset.json
```

The test suite contains:

- Deterministic guardrail tests for PII, schema failures, fallbacks, and safe
  memory behavior.
- Retrieval tests for all dataset cases. These tests do not call a chat LLM,
  but they use the configured embedding provider. If the local FAISS index is
  missing, the tests build it once from `app/data/docs`.
- Answer keyword tests for six representative positive cases. These tests call
  the provider and model configured by `TEST_LLM_PROVIDER` and
  `TEST_LLM_MODEL`.

For the default local setup, start Ollama and make sure both models are present:

```bash
ollama serve
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text:latest
```

Install dependencies and run the tests from the backend folder:

```bash
cd backend
pip install -r requirements.txt

pytest tests/test_guardrails.py -v
pytest tests/test_retrieval.py -v
pytest -m llm -v
pytest -m "not llm" -v
```

Live-LLM tests use a unique session ID and isolated in-memory test state for
every case. They do not write to or clear the existing SQLite memory database
or chat logs.

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
