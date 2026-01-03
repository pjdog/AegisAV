/**
 * ChatPanel - Real-time chat for discussing decisions during scenario runs
 *
 * Features:
 * - Send and receive messages in real-time via WebSocket
 * - Reference specific decisions in messages
 * - System messages for agent decisions
 * - Scrollable message history
 */

import React, { useState, useEffect, useRef, useCallback } from "react";

export interface ChatMessage {
  id: string;
  timestamp: string;
  message_type: "user" | "system" | "agent";
  sender: string;
  content: string;
  run_id?: string;
  decision_id?: string;
  metadata?: Record<string, unknown>;
}

interface Props {
  wsConnected: boolean;
  runId?: string;
  onDecisionClick?: (decisionId: string) => void;
  /** External messages from parent (via WebSocket) */
  externalMessages?: ChatMessage[];
}

export const ChatPanel: React.FC<Props> = ({
  wsConnected,
  runId,
  onDecisionClick,
  externalMessages = [],
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [senderName, setSenderName] = useState("User");
  const [isLoading, setIsLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Fetch initial messages
  const fetchMessages = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: "100" });
      if (runId) {
        params.set("run_id", runId);
      } else {
        params.set("current_run", "true");
      }
      const resp = await fetch(`/api/chat/messages?${params}`);
      if (resp.ok) {
        const data = await resp.json();
        setMessages(data.messages || []);
      }
    } catch (err) {
      console.error("Failed to fetch messages:", err);
    }
  }, [runId]);

  useEffect(() => {
    fetchMessages();
  }, [fetchMessages]);

  // Merge external messages from parent WebSocket
  useEffect(() => {
    if (externalMessages.length === 0) return;
    setMessages((prev) => {
      const newMessages = externalMessages.filter(
        (ext) => !prev.some((m) => m.id === ext.id)
      );
      if (newMessages.length === 0) return prev;
      return [...prev, ...newMessages].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
    });
  }, [externalMessages]);

  // Listen for WebSocket chat messages (fallback for direct window messages)
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.event_type === "chat_message" && data.message) {
          setMessages((prev) => {
            // Avoid duplicates
            const exists = prev.some((m) => m.id === data.message.id);
            if (exists) return prev;
            return [...prev, data.message];
          });
        }
      } catch {
        // Ignore non-JSON messages
      }
    };

    // Find the WebSocket connection and add listener
    // This assumes the WebSocket is managed at the Dashboard level
    // and events are dispatched globally
    window.addEventListener("message", handleMessage);

    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, []);

  // Send a message
  const sendMessage = async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    try {
      const resp = await fetch("/api/chat/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: trimmed,
          sender: senderName || "User",
        }),
      });

      if (resp.ok) {
        const data = await resp.json();
        // Add message immediately (WebSocket will also broadcast it)
        setMessages((prev) => {
          const exists = prev.some((m) => m.id === data.message?.id);
          if (exists) return prev;
          return [...prev, data.message];
        });
        setInputValue("");
      } else {
        console.error("Failed to send message:", resp.statusText);
      }
    } catch (err) {
      console.error("Failed to send message:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  };

  const getMessageTypeClass = (type: string) => {
    switch (type) {
      case "agent":
        return "chat-message-agent";
      case "system":
        return "chat-message-system";
      default:
        return "chat-message-user";
    }
  };

  const renderMessageContent = (content: string) => {
    // Simple markdown-like formatting for **bold**
    const parts = content.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-title">
          <span className="chat-icon">&#128172;</span>
          <span>Discussion</span>
          <span
            className={`chat-status ${wsConnected ? "connected" : "disconnected"}`}
            title={wsConnected ? "Connected" : "Disconnected"}
          />
        </div>
        <button
          className="chat-settings-btn"
          onClick={() => setShowSettings(!showSettings)}
          title="Settings"
        >
          &#9881;
        </button>
      </div>

      {/* Settings dropdown */}
      {showSettings && (
        <div className="chat-settings">
          <label className="chat-settings-label">
            Display Name:
            <input
              type="text"
              value={senderName}
              onChange={(e) => setSenderName(e.target.value)}
              placeholder="Your name"
              className="chat-settings-input"
              maxLength={50}
            />
          </label>
        </div>
      )}

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p className="chat-empty-title">No messages yet</p>
            <p className="chat-empty-desc">
              Start a conversation to discuss decisions and observations during the run.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`chat-message ${getMessageTypeClass(msg.message_type)}`}
            >
              <div className="chat-message-header">
                <span className="chat-sender">{msg.sender}</span>
                <span className="chat-time">{formatTime(msg.timestamp)}</span>
              </div>
              <div className="chat-content">
                {renderMessageContent(msg.content)}
              </div>
              {msg.decision_id && onDecisionClick && (
                <button
                  className="chat-decision-link"
                  onClick={() => onDecisionClick(msg.decision_id!)}
                >
                  View Decision
                </button>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-container">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          className="chat-input"
          disabled={isLoading}
          maxLength={2000}
        />
        <button
          onClick={sendMessage}
          disabled={!inputValue.trim() || isLoading}
          className="chat-send-btn"
          title="Send message"
        >
          {isLoading ? (
            <span className="chat-loading">...</span>
          ) : (
            <span className="chat-send-icon">&#10148;</span>
          )}
        </button>
      </div>
    </div>
  );
};

export default ChatPanel;
