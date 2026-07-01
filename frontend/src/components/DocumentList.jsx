import { useState } from 'react'
import { deleteDocument } from '../services/api'

function formatDate(isoString) {
  if (!isoString) return '—'
  const date = new Date(isoString)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString()
}

function DocumentList({ documents, isLoading, error, onRefresh, onDeleted }) {
  const [deletingId, setDeletingId] = useState(null)
  const [deleteError, setDeleteError] = useState(null)

  async function handleDelete(docId) {
    setDeletingId(docId)
    setDeleteError(null)
    try {
      await deleteDocument(docId)
      onDeleted()
    } catch (err) {
      setDeleteError(err.message)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <section className="document-list">
      <div className="document-list-header">
        <h2>Uploaded Documents</h2>
        <div className="document-list-actions">
          <button onClick={onRefresh} disabled={isLoading}>
            Refresh
          </button>
          <button disabled title="Bulk delete is not yet supported by the backend">
            Clear All
          </button>
        </div>
      </div>

      {error && <p className="error-message">{error}</p>}
      {deleteError && <p className="error-message">{deleteError}</p>}
      {isLoading && <p className="status-message">Loading documents…</p>}

      {!isLoading && documents.length === 0 && !error && (
        <p className="empty-message">No documents uploaded yet.</p>
      )}

      {documents.length > 0 && (
        <table className="document-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Uploaded</th>
              <th>Pages</th>
              <th>Chunks</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr key={doc.doc_id}>
                <td>{doc.filename}</td>
                <td>{formatDate(doc.uploaded_at)}</td>
                <td>{doc.pages ?? '—'}</td>
                <td>{doc.chunks ?? '—'}</td>
                <td>
                  <button
                    className="delete-button"
                    onClick={() => handleDelete(doc.doc_id)}
                    disabled={deletingId === doc.doc_id}
                  >
                    {deletingId === doc.doc_id ? 'Deleting…' : 'Delete'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

export default DocumentList