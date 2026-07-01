import { useCallback, useEffect, useState } from 'react'
import UploadPanel from './components/UploadPanel'
import DocumentList from './components/DocumentList'
import ChatWindow from './components/ChatWindow'
import { getDocuments } from './services/api'
import './styles.css'

const TABS = ['Documents', 'Ask Questions', 'Insights']

function App() {
  const [activeTab, setActiveTab] = useState('Documents')
  const [documents, setDocuments] = useState([])
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false)
  const [documentsError, setDocumentsError] = useState(null)

  const refreshDocuments = useCallback(async () => {
    setIsLoadingDocuments(true)
    setDocumentsError(null)
    try {
      const docs = await getDocuments()
      setDocuments(docs ?? [])
    } catch (err) {
      setDocumentsError(err.message)
    } finally {
      setIsLoadingDocuments(false)
    }
  }, [])

  useEffect(() => {
    refreshDocuments()
  }, [refreshDocuments])

  return (
    <div className="app">
      <header className="app-header">
        <h1>Document Intelligence</h1>
      </header>

      <nav className="tab-bar">
        {TABS.map((tab) => {
          const isDisabled = tab === 'Insights'
          return (
            <button
              key={tab}
              className={`tab-button ${activeTab === tab ? 'tab-button-active' : ''}`}
              onClick={() => setActiveTab(tab)}
              disabled={isDisabled}
              title={isDisabled ? 'Coming soon' : undefined}
            >
              {tab}
            </button>
          )
        })}
      </nav>

      <main className="tab-content">
        {activeTab === 'Documents' && (
          <>
            <UploadPanel onUploaded={refreshDocuments} />
            <DocumentList
              documents={documents}
              isLoading={isLoadingDocuments}
              error={documentsError}
              onRefresh={refreshDocuments}
              onDeleted={refreshDocuments}
            />
          </>
        )}

        {activeTab === 'Ask Questions' && <ChatWindow />}
      </main>
    </div>
  )
}

export default App
