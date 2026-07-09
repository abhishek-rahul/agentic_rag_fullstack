# Frontend - Agentic RAG Chat UI

React + Vite frontend for the FastAPI backend.

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Usage

1. Start the backend on port 8000.
2. Click **Ingest Docs** once.
3. Select provider: Ollama or OpenAI.
4. Enter a model name.
5. Ask a question such as: `What is the refund policy?`

## Optional env

Create `frontend/.env` if your backend uses another URL:

```bash
VITE_API_BASE_URL=http://localhost:8000
```
