/**
 * Hand-curated subset of the FastAPI schema that the UI actually consumes.
 *
 * The full generated surface lives in ``types.gen.ts`` after
 * ``npm run gen-types`` is run against a live backend. This file is the
 * stable fallback so the bundle keeps compiling even when the generated
 * file is absent (fresh clone, CI without the backend, etc.).
 */

export interface Aggregate {
  wer_mean?: number | null
  cer_mean?: number | null
  mer_mean?: number | null
  wil_mean?: number | null
  rtfx_mean?: number | null
  rtfx_p95?: number | null
  vram_peak_mb?: number | null
  wall_time_s?: number | null
  word_count?: number | null
  wer_ci_lower?: number | null
  wer_ci_upper?: number | null
}

export interface Run {
  run_id: string
  model_id: string
  backend: string
  lang: string
  status: string
  label?: string | null
  params?: Record<string, unknown> | null
  aggregate?: Aggregate | null
  created_at?: string
}

export interface Segment {
  offset_s: number
  duration_s: number
  ref_text: string
  hyp_text: string
  wer?: number | null
  rtfx?: number | null
}

export interface CompareRunEntry {
  run_id: string
  params: Record<string, unknown>
  aggregate: Record<string, number | null>
  is_baseline: boolean
  delta_wer_mean?: number | null
  delta_cer_mean?: number | null
  delta_mer_mean?: number | null
  delta_rtfx_mean?: number | null
  delta_rtfx_p95?: number | null
}

export interface CompareResponse {
  runs: CompareRunEntry[]
  params_diff: string[]
  params_same: string[]
  wilcoxon_p?: number | null
}

export interface VRAMSnapshot {
  type?: string
  available: boolean
  used_mb: number
  total_mb: number
  free_mb?: number
  pct: number
}

export interface ActivityRecord {
  ts?: string
  level: string
  source?: string
  message: string
  [extra: string]: unknown
}

export interface Model {
  model_id: string
  family: string
  name: string
  backend: string
  local_path: string
  default_params?: Record<string, unknown> | null
  loaded?: boolean
}

export interface Dataset {
  dataset_id: string
  name: string
  lang: string
  split: string
  source: string
  duration_s?: number | null
  verified: boolean
}

export interface RunProgressEvent {
  type: string
  run_id?: string
  status?: string
  segments_done?: number
  total_segments?: number
  elapsed_s?: number
  wer_mean?: number
  rtfx_mean?: number
  wall_time_s?: number
  vram_peak_mb?: number | null
  error?: string
}
