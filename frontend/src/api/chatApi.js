const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function sendChatMessage({ message, session_id, provider, model }) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      session_id,
      provider,
      model,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || "Chat request failed");
  }

  return response.json();
}

export async function ingestDocs() {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: "POST",
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || "Ingest request failed");
  }

  return response.json();
}
