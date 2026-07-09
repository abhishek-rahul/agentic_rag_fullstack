import MessageBubble from "./MessageBubble";

export default function ChatPanel({
  title,
  sessionId,
  provider,
  model,
  messages,
  loading,
  onProviderChange,
  onModelChange,
}) {
  return (
    <section className="chat-panel">
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          <p>
            {provider} / {model}
          </p>
        </div>
        <span className="panel-badge">{loading ? "Loading" : "Ready"}</span>
      </div>

      <div className="panel-controls">
        <label>
          Provider
          <select value={provider} onChange={(event) => onProviderChange(event.target.value)}>
            <option value="ollama">Ollama</option>
            <option value="openai">OpenAI</option>
          </select>
        </label>

        <label>
          Model
          <input value={model} onChange={(event) => onModelChange(event.target.value)} />
        </label>
      </div>

      <div className="selected-model">
        <span>Session: {sessionId}</span>
      </div>

      <div className="messages-panel">
        {messages.length === 0 && (
          <div className="empty-state">
            <h3>No messages yet.</h3>
            <p>Ask a shared question to compare this model.</p>
          </div>
        )}

        {messages.map((message, index) => (
          <MessageBubble key={`${title}-${message.role}-${index}`} message={message} />
        ))}

        {loading && <div className="loading">Assistant is thinking...</div>}
      </div>
    </section>
  );
}
