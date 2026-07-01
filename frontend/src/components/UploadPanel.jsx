import { useRef, useState } from 'react'
import { uploadFiles } from '../services/api'

function UploadPanel({ onUploaded }) {
  const inputRef = useRef(null)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)

  function handleFileChange(event) {
    setSelectedFiles(Array.from(event.target.files))
    setResults([])
    setError(null)
  }

  async function handleUpload() {
    if (selectedFiles.length === 0 || isUploading) return

    setIsUploading(true)
    setError(null)
    try {
      const uploadResults = await uploadFiles(selectedFiles)
      setResults(uploadResults ?? [])
      setSelectedFiles([])
      if (inputRef.current) inputRef.current.value = ''
      onUploaded()
    } catch (err) {
      setError(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <section className="upload-panel">
      <h2>Upload Documents</h2>

      <div className="upload-controls">
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          onChange={handleFileChange}
          disabled={isUploading}
        />
        <button onClick={handleUpload} disabled={isUploading || selectedFiles.length === 0}>
          {isUploading
            ? 'Uploading…'
            : selectedFiles.length > 0
              ? `Upload ${selectedFiles.length} file${selectedFiles.length === 1 ? '' : 's'}`
              : 'Upload'}
        </button>
      </div>

      {isUploading && <p className="status-message">Uploading and processing files…</p>}
      {error && <p className="error-message">{error}</p>}

      {results.length === 0 && !isUploading && !error ? null : (
        <ul className="upload-results">
          {results.map((result, index) => (
            <li
              key={`${result.filename}-${index}`}
              className={result.status === 'success' ? 'result-success' : 'result-failure'}
            >
              <span className="result-filename">{result.filename || '(unnamed file)'}</span>
              {result.status === 'success' ? (
                <span className="result-detail">
                  {result.pages ?? '—'} page{result.pages === 1 ? '' : 's'} ·{' '}
                  {result.chunks ?? '—'} chunk{result.chunks === 1 ? '' : 's'}
                </span>
              ) : (
                <span className="result-detail">Rejected — {result.reason || 'unknown reason'}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default UploadPanel