export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const bubbleClassName = [
    "message-bubble",
    isUser ? "user-bubble" : "assistant-bubble",
    message.isError ? "error-bubble" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={bubbleClassName}>
        <div className="message-role">{isUser ? "You" : "Assistant"}</div>
        <div className="message-content">{message.content}</div>

        {!isUser && message.sources?.length > 0 && (
          <div className="sources">
            <div className="sources-title">Sources</div>
            {message.sources.map((source, index) => (
              <div className="source-item" key={`${source.source}-${index}`}>
                <strong>{source.source || "Local document"}</strong>
                <p>{source.content_preview}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
