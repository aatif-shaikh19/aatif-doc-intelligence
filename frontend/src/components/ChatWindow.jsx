import { useEffect, useRef, useState } from 'react'
import { askQuestion } from '../services/api'
import CitationCard from './CitationCard'

function ChatWindow() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [error, setError] = useState(null)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isAsking])

  async function handleSubmit() {
    const trimmed = question.trim()
    if (!trimmed || isAsking) return

    const userMessage = { id: `${Date.now()}-user`, role: 'user', text: trimmed }
    setMessages((prev) => [...prev, userMessage])
    setQuestion('')
    setIsAsking(true)
    setError(null)

    try {
      const result = await askQuestion(trimmed)
      const assistantMessage = {
        id: `${Date.now()}-assistant`,
        role: 'assistant',
        text: result.answer || 'No answer was returned.',
        confidence: result.confidence ?? 0,
        citations: result.citations ?? [],
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      setError(err.message)
    } finally {
      setIsAsking(false)
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  return (
    <section className="chat-window">
      <div className="chat-history">
        {messages.length === 0 && (
          <p className="empty-message">Ask a question about your uploaded documents.</p>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`chat-message chat-message-${message.role}`}>
            <div className="chat-bubble">
              <p>{message.text}</p>
              {message.role === 'assistant' && (
                <>
                  <p className="confidence-label">
                    Confidence: {Math.round(message.confidence * 100)}%
                  </p>
                  {message.citations.length > 0 && (
                    <div className="citation-list">
                      {message.citations.map((citation, index) => (
                        <CitationCard
                          key={`${message.id}-citation-${index}`}
                          citation={citation}
                        />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {isAsking && (
          <div className="chat-message chat-message-assistant">
            <div className="chat-bubble">
              <span className="spinner" role="status" aria-label="Waiting for answer" />
            </div>
          </div>
        )}

        <div ref={endRef} />
      </div>

      {error && <p className="error-message">{error}</p>}

      <div className="chat-input-row">
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          disabled={isAsking}
        />
        <button onClick={handleSubmit} disabled={isAsking || !question.trim()}>
          {isAsking ? 'Asking…' : 'Ask'}
        </button>
      </div>
    </section>
  )
}

export default ChatWindow