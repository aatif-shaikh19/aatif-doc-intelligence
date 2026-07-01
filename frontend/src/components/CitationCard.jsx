import { useState } from 'react'

function CitationCard({ citation }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className="citation-card">
      <button
        type="button"
        className="citation-toggle"
        onClick={() => setIsExpanded((prev) => !prev)}
        aria-expanded={isExpanded}
      >
        <span className="citation-source">
          {citation.filename} · page {citation.page_number}
        </span>
        <span className="citation-toggle-icon">{isExpanded ? '−' : '+'}</span>
      </button>
      {isExpanded && <p className="citation-excerpt">{citation.excerpt}</p>}
    </div>
  )
}

export default CitationCard