# Backend - Local Agentic RAG Assistant

FastAPI backend with LangChain, LangGraph, FAISS, SQLite memory, OpenAI support, and Ollama support.

This backend includes optional request and RAG-result guardrails, deterministic
regression tests, and isolated DeepEval, Ragas, and Promptfoo evaluations.

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

## Steps 9-11: RAG evaluations

The same dataset at `app/data/eval_dataset.json` drives every evaluation. Four
cases marked `run_core_eval` contain concise reference answers and are used by
DeepEval, Ragas, and Promptfoo.

### Environment separation

Use the existing `.venv` for the application, retrieval, record generation,
and Promptfoo's Python provider. Use `.venv-evals` only for DeepEval/Ragas and
judge-model calls. DeepEval and Ragas read normalized JSON records and never
import the production RAG services.

Create the judge environment from the backend directory:

```powershell
py -3.13 -m venv .venv-evals
.\.venv-evals\Scripts\python.exe -m pip install --upgrade pip
.\.venv-evals\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv-evals\Scripts\python.exe -m pip check
```

### Generate four shared records

Start the configured embedding and generation providers first. If FAISS files
are missing, this command ingests the current documents once. It uses ephemeral
memory and no-op chat logging, so it does not modify `memory.db` or
`chat_runs.jsonl`.

```powershell
.\.venv\Scripts\python.exe -m evals.generate_records
```

The Git-ignored output is
`eval_results/shared/core_rag_records.json`. A guardrail-passed record may store
the raw answer. Every blocked record stores `raw_answer: null`, only the safe
fallback/reason, and no `llm_response`.

### DeepEval

DeepEval reads the four frozen records and calls only the judge model. It checks
Faithfulness and Answer Relevancy with threshold `0.6`.

```powershell
.\.venv-evals\Scripts\deepeval.exe test run evals/deepeval/test_rag_metrics.py -v
```

The default judge is `gpt-4o-mini` and requires `OPENAI_API_KEY`. No cloud report
upload is configured. Running the judge sends the four selected questions,
generated answers, and retrieved context to the configured OpenAI judge.

### Ragas

Ragas reads the same records, calls only the judge, and calculates Faithfulness
and Context Precision. Results are written locally as JSON and CSV.
Its judge call sends the stored questions, answers, contexts, and reference
answers to the configured OpenAI model.

```powershell
.\.venv-evals\Scripts\python.exe -m evals.ragas.run_ragas
```

Use `--case wfh-001` for a one-case smoke run.

### Promptfoo

Promptfoo calls the actual application flow rather than calling an LLM provider
directly. Its default run is four cases times two prompts times one local Ollama
model, for eight application calls.

```powershell
cd evals\promptfoo
npm install
$env:PROMPTFOO_PYTHON = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
npm run validate
npm run eval:local
```

The optional Ollama plus OpenAI comparison performs sixteen calls:

```powershell
npm run eval:all
```

Promptfoo uses no cache or cloud sharing. Its JSON and HTML reports are stored
under `eval_results/promptfoo`.

### Complete verification

```powershell
.\.venv\Scripts\python.exe -m compileall -x "node_modules|\.promptfoo" app tests evals
.\.venv\Scripts\python.exe -m pytest tests/test_eval_runner.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_guardrails.py -v
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -v
.\.venv\Scripts\python.exe -m pytest -m "not llm" -v
.\.venv\Scripts\python.exe -m pytest -m llm -v
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
