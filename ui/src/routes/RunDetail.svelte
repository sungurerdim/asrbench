<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { api } from '../lib/api'
  import { connectRun } from '../lib/ws'
  import type { Run, RunProgressEvent, Segment } from '../lib/types'

  interface Props {
    runId: string
  }

  let { runId }: Props = $props()

  let run = $state<Run | null>(null)
  let segments = $state<Segment[]>([])
  let progress = $state<{ done: number; total: number; pct: number }>({
    done: 0,
    total: 0,
    pct: 0,
  })
  let liveWs: WebSocket | null = null
  let error = $state<string>('')

  onMount(async () => {
    try {
      run = await api.getRun(runId)
      segments = await api.getSegments(runId)
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load run'
    }

    if (run?.status === 'running' || run?.status === 'pending') {
      liveWs = connectRun(runId, (msg: RunProgressEvent) => {
        if (msg.type === 'segment_done') {
          const done = msg.segments_done ?? 0
          const total = msg.total_segments ?? 0
          progress = {
            done,
            total,
            pct: total > 0 ? Math.round((done / total) * 100) : 0,
          }
        } else if (msg.type === 'complete' || msg.status === 'completed') {
          progress.pct = 100
          api.getRun(runId).then((r) => {
            run = r
          })
        }
      })
    }
  })

  onDestroy(() => liveWs?.close())

  async function cancel(): Promise<void> {
    try {
      await api.cancelRun(runId)
    } catch (e) {
      error = e instanceof Error ? e.message : 'Cancel failed'
    }
  }
</script>

<h2>Run Detail — <code>{runId?.slice(0, 8)}…</code></h2>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

{#if run?.status === 'running' || run?.status === 'pending'}
  <div style="margin:1rem 0;">
    <div style="background:#313244;border-radius:4px;height:20px;width:100%;">
      <div
        style="background:#89b4fa;height:100%;width:{progress.pct}%;border-radius:4px;transition:width 0.3s;"
      ></div>
    </div>
    <small>{progress.done}/{progress.total} segments ({progress.pct}%)</small>
    <button
      onclick={cancel}
      style="margin-left:1rem;padding:0.25rem 0.75rem;background:#f38ba8;color:#1e1e2e;border:none;border-radius:4px;"
    >
      Cancel
    </button>
  </div>
{/if}

{#if run?.aggregate}
  <div
    style="display:flex;gap:2rem;margin:1rem 0;padding:1rem;background:#313244;border-radius:8px;"
  >
    <span>WER: <strong>{run.aggregate.wer_mean?.toFixed(3) ?? '—'}</strong></span>
    <span>CER: <strong>{run.aggregate.cer_mean?.toFixed(3) ?? '—'}</strong></span>
    <span>RTFx: <strong>{run.aggregate.rtfx_mean?.toFixed(1) ?? '—'}×</strong></span>
    <span>VRAM: <strong>{run.aggregate.vram_peak_mb?.toFixed(0) ?? '—'} MB</strong></span>
    <span>Time: <strong>{run.aggregate.wall_time_s?.toFixed(1) ?? '—'}s</strong></span>
  </div>
{/if}

<h3>Segments</h3>
<table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
  <thead>
    <tr style="color:#89b4fa;text-align:left;">
      <th>Offset</th><th>Duration</th><th>Reference</th><th>Hypothesis</th>
    </tr>
  </thead>
  <tbody>
    {#each segments as seg, i (i)}
      <tr style="border-top:1px solid #313244;">
        <td style="padding:0.3rem 0;">{seg.offset_s.toFixed(1)}s</td>
        <td>{seg.duration_s.toFixed(1)}s</td>
        <td
          style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
          title={seg.ref_text}
        >
          {seg.ref_text}
        </td>
        <td
          style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
          title={seg.hyp_text}
        >
          {seg.hyp_text}
        </td>
      </tr>
    {/each}
  </tbody>
</table>

<div style="margin-top:1rem;display:flex;gap:0.5rem;">
  <a href={api.exportRun(runId, 'json')} download>Export JSON</a>
  <a href={api.exportRun(runId, 'csv')} download>Export CSV</a>
</div>
