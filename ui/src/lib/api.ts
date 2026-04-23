/** Typed API client for the asrbench FastAPI backend. */

import type {
  CompareResponse,
  Dataset,
  Model,
  Run,
  Segment,
} from './types'

const BASE = ''

interface ApiError {
  detail: string
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const opts: RequestInit = {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  }
  const res = await fetch(BASE + path, opts)
  if (!res.ok) {
    const detail = (await res
      .json()
      .catch(() => ({ detail: res.statusText }))) as ApiError
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  if (res.status === 204) return null as T
  return (await res.json()) as T
}

export const api = {
  // System
  health: () => request<{ status: string; version: string }>('GET', '/system/health'),
  vram: () =>
    request<{ gpu_available: boolean; gpus: Array<Record<string, unknown>> }>(
      'GET',
      '/system/vram',
    ),

  // Runs
  listRuns: (params: Record<string, string | number> = {}) => {
    const q = new URLSearchParams(
      Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])),
    ).toString()
    return request<Run[]>('GET', `/runs${q ? '?' + q : ''}`)
  },
  startRun: (body: Record<string, unknown>) =>
    request<{ run_id: string; status: string }>('POST', '/runs/start', body),
  getRun: (id: string) => request<Run>('GET', `/runs/${id}`),
  deleteRun: (id: string) => request<null>('DELETE', `/runs/${id}`),
  cancelRun: (id: string) =>
    request<{ run_id: string; status: string }>('POST', `/runs/${id}/cancel`),
  retryRun: (id: string) =>
    request<{ original_run_id: string; new_run_id: string; status: string }>(
      'POST',
      `/runs/${id}/retry`,
    ),
  getSegments: (id: string, page = 1) =>
    request<Segment[]>('GET', `/runs/${id}/segments?page=${page}`),
  compareRuns: (ids: string[]) =>
    request<CompareResponse>('GET', `/runs/compare?ids=${ids.join(',')}`),
  exportRun: (id: string, fmt: 'json' | 'csv' = 'json') =>
    `${BASE}/runs/${id}/export?fmt=${fmt}`,

  // Models
  listModels: () => request<Model[]>('GET', '/models'),
  registerModel: (body: Record<string, unknown>) =>
    request<{ model_id: string; name: string }>('POST', '/models', body),
  loadModel: (id: string) =>
    request<{ model_id: string; vram_used_mb: number; vram_total_mb: number }>(
      'POST',
      `/models/${id}/load`,
    ),
  unloadModel: (id: string) =>
    request<{ model_id: string }>('POST', `/models/${id}/unload`),

  // Datasets
  listDatasets: (params: Record<string, string> = {}) => {
    const q = new URLSearchParams(params).toString()
    return request<Dataset[]>('GET', `/datasets${q ? '?' + q : ''}`)
  },
  fetchDataset: (body: Record<string, unknown>) =>
    request<{
      dataset_id: string
      name: string
      status: string
      stream_url: string
    }>('POST', '/datasets/fetch', body),
  deleteDataset: (id: string, deleteFiles = false) =>
    request<null>(
      'DELETE',
      `/datasets/${id}${deleteFiles ? '?delete_files=true' : ''}`,
    ),
}
