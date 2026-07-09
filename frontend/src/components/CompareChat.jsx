import { useMemo, useState } from "react";
import { ingestDocs, sendChatMessage } from "../api/chatApi";
import ChatPanel from "./ChatPanel";

const DEFAULT_CONFIGS = {
  modelA: {
    provider: "ollama",
    model: "qwen2.5:0.5b",
  },
  modelB: {
    provider: "ollama",
    model: "llama3.1",
  },
};

function createSessionId() {
  if (globalThis.crypto?.randomUUID) {
    return `session-${globalThis.crypto.randomUUID()}`;
  }

  return `session-${Date.now()}`;
}

function defaultModelForProvider(provider) {
  return provider === "openai" ? "gpt-4o-mini" : "qwen2.5:0.5b";
}

function createAssistantMessage(result) {
  return {
    role: "assistant",
    content: result.answer || "No answer returned.",
    sources: result.sources || [],
  };
}

function createErrorMessage(error) {
  return {
    role: "assistant",
    content: `Error: ${error.message}`,
    sources: [],
    isError: true,
  };
}

export default function CompareChat() {
  const [baseSessionId, setBaseSessionId] = useState(() => createSessionId());
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("");
  const [modelAConfig, setModelAConfig] = useState(DEFAULT_CONFIGS.modelA);
  const [modelBConfig, setModelBConfig] = useState(DEFAULT_CONFIGS.modelB);
  const [modelAMessages, setModelAMessages] = useState([]);
  const [modelBMessages, setModelBMessages] = useState([]);
  const [modelALoading, setModelALoading] = useState(false);
  const [modelBLoading, setModelBLoading] = useState(false);

  const sessionIds = useMemo(() => {
    return {
      modelA: `${baseSessionId}-model-a`,
      modelB: `${baseSessionId}-model-b`,
    };
  }, [baseSessionId]);

  const isSending = modelALoading || modelBLoading;
  const placeholder = "Ask once, compare both model answers...";

  function updateModelAConfig(nextConfig) {
    setModelAConfig((current) => ({ ...current, ...nextConfig }));
  }

  function updateModelBConfig(nextConfig) {
    setModelBConfig((current) => ({ ...current, ...nextConfig }));
  }

  function handleProviderChange(panel, provider) {
    const nextConfig = {
      provider,
      model: defaultModelForProvider(provider),
    };

    if (panel === "modelA") {
      updateModelAConfig(nextConfig);
      return;
    }

    updateModelBConfig(nextConfig);
  }

  async function handleIngest() {
    setStatus("Ingesting local documents...");

    try {
      const result = await ingestDocs();
      setStatus(`Ingest complete: ${result.documents_loaded} docs, ${result.chunks_created} chunks`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function startNewSession() {
    setBaseSessionId(createSessionId());
    setModelAMessages([]);
    setModelBMessages([]);
    setStatus("New comparison session created.");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const trimmed = input.trim();
    if (!trimmed || isSending) return;

    const userMessage = { role: "user", content: trimmed };
    setModelAMessages((previous) => [...previous, userMessage]);
    setModelBMessages((previous) => [...previous, userMessage]);
    setInput("");
    setStatus("");
    setModelALoading(true);
    setModelBLoading(true);

    const modelARequest = sendChatMessage({
      message: trimmed,
      session_id: sessionIds.modelA,
      provider: modelAConfig.provider,
      model: modelAConfig.model,
    })
      .then((result) => {
        setModelAMessages((previous) => [...previous, createAssistantMessage(result)]);
      })
      .catch((error) => {
        setModelAMessages((previous) => [...previous, createErrorMessage(error)]);
      })
      .finally(() => {
        setModelALoading(false);
      });

    const modelBRequest = sendChatMessage({
      message: trimmed,
      session_id: sessionIds.modelB,
      provider: modelBConfig.provider,
      model: modelBConfig.model,
    })
      .then((result) => {
        setModelBMessages((previous) => [...previous, createAssistantMessage(result)]);
      })
      .catch((error) => {
        setModelBMessages((previous) => [...previous, createErrorMessage(error)]);
      })
      .finally(() => {
        setModelBLoading(false);
      });

    await Promise.allSettled([modelARequest, modelBRequest]);
  }

  return (
    <div className="chat-shell">
      <header className="app-header">
        <div>
          <h1>Agentic RAG Comparison</h1>
          <p>Ask once and compare two model responses side by side.</p>
        </div>
        <div className="header-actions">
          <button className="secondary-button" onClick={handleIngest}>
            Ingest Docs
          </button>
          <button className="secondary-button" onClick={startNewSession}>
            New Session
          </button>
        </div>
      </header>

      <section className="session-card">
        <label>
          Base Session ID
          <input value={baseSessionId} onChange={(event) => setBaseSessionId(event.target.value)} />
        </label>
        <div className="session-id-list">
          <span>Model A: {sessionIds.modelA}</span>
          <span>Model B: {sessionIds.modelB}</span>
        </div>
      </section>

      {status && <div className="status-bar">{status}</div>}

      <main className="comparison-grid">
        <ChatPanel
          title="Model A"
          sessionId={sessionIds.modelA}
          provider={modelAConfig.provider}
          model={modelAConfig.model}
          messages={modelAMessages}
          loading={modelALoading}
          onProviderChange={(provider) => handleProviderChange("modelA", provider)}
          onModelChange={(model) => updateModelAConfig({ model })}
        />
        <ChatPanel
          title="Model B"
          sessionId={sessionIds.modelB}
          provider={modelBConfig.provider}
          model={modelBConfig.model}
          messages={modelBMessages}
          loading={modelBLoading}
          onProviderChange={(provider) => handleProviderChange("modelB", provider)}
          onModelChange={(model) => updateModelBConfig({ model })}
        />
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <input value={input} onChange={(event) => setInput(event.target.value)} placeholder={placeholder} />
        <button type="submit" disabled={isSending}>
          Send
        </button>
      </form>
    </div>
  );
}
