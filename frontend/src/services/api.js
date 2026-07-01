import axios from 'axios'

const API_BASE_URL = 'http://localhost:8001'

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

function toApiError(error) {
  if (error.code === 'ECONNABORTED') {
    return new Error('The request timed out. Please try again.')
  }
  if (error.response) {
    const data = error.response.data
    const detail = data?.detail || data?.reason || data?.message
    return new Error(detail || `Request failed (${error.response.status})`)
  }
  if (error.request) {
    return new Error('Could not reach the backend. Is the server running?')
  }
  return new Error(error.message || 'Something went wrong.')
}

export async function uploadFiles(files) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  try {
    const response = await client.post('/upload', formData, { timeout: 120000 })
    return response.data.results
  } catch (error) {
    throw toApiError(error)
  }
}

export async function getDocuments() {
  try {
    const response = await client.get('/documents')
    return response.data.documents
  } catch (error) {
    throw toApiError(error)
  }
}

export async function deleteDocument(id) {
  try {
    const response = await client.delete(`/documents/${id}`)
    return response.data
  } catch (error) {
    throw toApiError(error)
  }
}

export async function askQuestion(question) {
  try {
    const response = await client.post('/query', { question })
    return response.data
  } catch (error) {
    throw toApiError(error)
  }
}