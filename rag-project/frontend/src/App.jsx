import { useState, useEffect, useRef } from 'react';
import './App.css';

// The FastAPI backend base URL
const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  // -------------------------------------------------------------------------
  // 1. React States (What the component remembers)
  // -------------------------------------------------------------------------
  // `messages`: List of chat turns ({ role: 'user' | 'assistant', content, sources, etc. })
  const [messages, setMessages] = useState([]);
  
  // `input`: The text currently typed into the input field
  const [input, setInput] = useState('');
  
  // `isLoading`: Boolean to show loading spinner while waiting for LLM + Retrieval
  const [isLoading, setIsLoading] = useState(false);
  
  // `backendStatus`: 'checking' | 'online' | 'offline'
  const [backendStatus, setBackendStatus] = useState('checking');
  
  // `topK`: Number of document chunks to retrieve (defaults to 3)
  const [topK, setTopK] = useState(3);
  
  // `activeSourcesIndex`: Tracks which message has its sources drawer open
  const [openSourcesIndex, setOpenSourcesIndex] = useState(null);

  // Ref to automatically scroll the chat container to the bottom
  const chatEndRef = useRef(null);

  // -------------------------------------------------------------------------
  // 2. Health Check Effect (Check if FastAPI server is running)
  // -------------------------------------------------------------------------
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`);
        if (res.ok) {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch (err) {
        setBackendStatus('offline');
      }
    };

    checkBackend();
    // Poll health every 10 seconds to auto-detect when backend starts
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom whenever messages or loading state changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // -------------------------------------------------------------------------
  // 3. Handlers
  // -------------------------------------------------------------------------
  const handleSend = async (questionText = input) => {
    const textToSend = questionText.trim();
    if (!textToSend || isLoading) return;

    // 1. Immediately append user's question to the chat UI
    const newUserMessage = { role: 'user', content: textToSend };
    const updatedMessages = [...messages, newUserMessage];
    setMessages(updatedMessages);
    setInput('');
    setIsLoading(true);

    // 2. Build conversation history for query rewriter & context in RAG
    // Only pass role and content to the backend
    const conversationHistory = updatedMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      // 3. Call FastAPI POST /api/ask
      // Notice: fetch() makes an asynchronous HTTP request over the network.
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: textToSend,
          top_k: topK,
          conversation: conversationHistory,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${response.status}`);
      }

      // 4. Parse the JSON response returned by rag.ask()
      const data = await response.json();

      // 5. Add assistant's answer + sources + metadata to the message list
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          searchQuery: data.search_query,
          citationValidation: data.citation_validation,
        },
      ]);
    } catch (error) {
      // Show user-friendly error in the chat
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Error: Could not get an answer. ${error.message}. Please ensure the FastAPI server is running on http://127.0.0.1:8000.`,
          isError: true,
          sources: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setOpenSourcesIndex(null);
  };

  // Sample starter questions to help the user test easily
  const samplePrompts = [
    "What is the main topic of the documents?",
    "Summarize the key findings from page 1.",
    "What are the challenges mentioned?",
  ];

  // -------------------------------------------------------------------------
  // 4. Render JSX (The UI)
  // -------------------------------------------------------------------------
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <div>
            <h1>RAG Assistant</h1>
            <p className="subtitle">Retrieval-Augmented Generation with FastAPI & React</p>
          </div>
        </div>

        <div className="header-actions">
          {/* Backend Connection Indicator */}
          <div className={`status-badge ${backendStatus}`}>
            <span className="dot"></span>
            <span>
              {backendStatus === 'online' && 'FastAPI Connected'}
              {backendStatus === 'offline' && 'Backend Offline (port 8000)'}
              {backendStatus === 'checking' && 'Checking Backend...'}
            </span>
          </div>

          {/* Top-K Selector */}
          <div className="topk-control">
            <label htmlFor="topk-select">Sources (k):</label>
            <select
              id="topk-select"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            >
              <option value={2}>2 chunks</option>
              <option value={3}>3 chunks</option>
              <option value={5}>5 chunks</option>
            </select>
          </div>

          {messages.length > 0 && (
            <button className="clear-btn" onClick={handleClearChat} title="Clear conversation">
              Clear
            </button>
          )}
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="chat-container">
        {messages.length === 0 ? (
          // Empty State / Welcome Screen
          <div className="welcome-card">
            <div className="welcome-icon">📚</div>
            <h2>Ask your Documents Anything</h2>
            <p>
              Your questions are rewritten for optimal search, matched against vector embeddings
              in FAISS, reranked, and answered using context-grounded citations.
            </p>

            <div className="suggestions">
              <span className="suggestions-title">Try asking:</span>
              <div className="chip-list">
                {samplePrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    className="suggestion-chip"
                    onClick={() => handleSend(prompt)}
                  >
                    "{prompt}"
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          // Message History
          <div className="messages-list">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`message-row ${msg.role === 'user' ? 'user' : 'assistant'}`}
              >
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>

                <div className="message-content">
                  <div className="message-bubble">
                    <p className="message-text">{msg.content}</p>
                  </div>

                  {/* Assistant Extra Info (Query rewrite, citations, sources) */}
                  {msg.role === 'assistant' && !msg.isError && (
                    <div className="meta-container">
                      {msg.searchQuery && (
                        <div className="meta-tag search-query">
                          <span className="meta-label">Query Rewritten:</span> "{msg.searchQuery}"
                        </div>
                      )}

                      {/* Sources Accordion */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="sources-wrapper">
                          <button
                            className="toggle-sources-btn"
                            onClick={() =>
                              setOpenSourcesIndex(openSourcesIndex === index ? null : index)
                            }
                          >
                            <span>
                              {openSourcesIndex === index ? '▼ Hide Sources' : '▶ View Sources'} ({msg.sources.length})
                            </span>
                          </button>

                          {openSourcesIndex === index && (
                            <div className="sources-grid">
                              {msg.sources.map((src, sIdx) => (
                                <div key={sIdx} className="source-card">
                                  <div className="source-header">
                                    <span className="source-doc">📄 {src.document || 'Document'}</span>
                                    <span className="source-page">Page {src.page ?? 'N/A'}</span>
                                  </div>
                                  <div className="source-scores">
                                    <span className="score-badge">
                                      Similarity: {(src.similarity * 100).toFixed(1)}%
                                    </span>
                                    {src.rerank_score !== undefined && src.rerank_score !== null && (
                                      <span className="score-badge rerank">
                                        Rerank: {Number(src.rerank_score).toFixed(3)}
                                      </span>
                                    )}
                                  </div>
                                  <div className="chunk-id">Chunk ID: {src.chunk_id}</div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="message-row assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="message-bubble loading-bubble">
                    <span className="pulse-dot"></span>
                    <span className="pulse-dot"></span>
                    <span className="pulse-dot"></span>
                    <span className="loading-label">Searching vector store & generating answer...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        )}
      </main>

      {/* Input Bar */}
      <footer className="input-section">
        <div className="input-wrapper">
          <textarea
            className="chat-input"
            rows="1"
            placeholder="Ask a question about your documents... (Press Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            className="send-button"
            onClick={() => handleSend()}
            disabled={!input.trim() || isLoading}
          >
            {isLoading ? (
              <span className="spinner"></span>
            ) : (
              <span>Send ➔</span>
            )}
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
