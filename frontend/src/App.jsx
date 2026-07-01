import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE_URL = 'http://localhost:8001'

function App() {
  const [status, setStatus] = useState('checking...')

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/health`)
      .then((response) => setStatus(response.data.status))
      .catch(() => setStatus('unreachable'))
  }, [])

  return (
    <div className="app">
      <h1>Document Intelligence</h1>
      <p>
        Backend status: <strong>{status}</strong>
      </p>
    </div>
  )
}

export default App
