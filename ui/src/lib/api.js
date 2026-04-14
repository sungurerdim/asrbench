/** Typed API client for asrbench FastAPI backend. */

const BASE = ''

async function request(method, path, body) {
  const opts = {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  }
  const res = await fetch(BASE + path, opts)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  // System
  health: () => request('GET', '/system/health'),
  vram: () => request('GET', '/system/vram'),

  // Runs
  listRuns: (params = {}) => {
    const q = new URLSearchParams(params).toString()
    return request('GET', `/runs${q ? '?' + q : ''}`)
  },
  startRun: (body) => request('POST', '/runs/start', body),
  getRun: (id) => request('GET', `/runs/${id}`),
  deleteRun: (id) => request('DELETE', `/runs/${id}`),
  cancelRun: (id) => request('POST', `/runs/${id}/cancel`),
  retryRun: (id) => request('POST', `/runs/${id}/retry`),
  getSegments: (id, page = 1) => request('GET', `/runs/${id}/segments?page=${page}`),
  compareRuns: (ids) => request('GET', `/runs/compare?ids=${ids.join(',')}`),
  exportRun: (id, fmt = 'json') => `${BASE}/runs/${id}/export?fmt=${fmt}`,

  // Models
  listModels: () => request('GET', '/models'),
  registerModel: (body) => request('POST', '/models', body),
  loadModel: (id) => request('POST', `/models/${id}/load`),
  unloadModel: (id) => request('POST', `/models/${id}/unload`),
  deleteModel: (id) => request('DELETE', `/models/${id}`),

  // Datasets
  listDatasets: (params = {}) => {
    const q = new URLSearchParams(params).toString()
    return request('GET', `/datasets${q ? '?' + q : ''}`)
  },
  fetchDataset: (body) => request('POST', '/datasets/fetch', body),
  deleteDataset: (id) => request('DELETE', `/datasets/${id}`),
}
