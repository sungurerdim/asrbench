<script>
  import { onMount, onDestroy } from 'svelte'
  import { api } from '../lib/api.js'
  import { connectRunLive } from '../lib/ws.js'

  let { runId } = $props()

  let run = $state(null)
  let segments = $state([])
  let progress = $state({ done: 0, total: 0, pct: 0, eta_s: 0 })
  let liveWs = null
  let error = $state('')

  onMount(async () => {
    try {
      run = await api.getRun(runId)
      segments = await api.getSegments(runId)
    } catch (e) {
      error = e.message
    }

    if (run?.status === 'running' || run?.status === 'pending') {
      liveWs = connectRunLive(runId, (msg) => {
        if (msg.type === 'segment') {
          progress = { done: msg.segments_done, total: msg.segments_total, pct: Math.round(msg.progress * 100), eta_s: msg.eta_s }
          segments = [msg.segment, ...segments].slice(0, 500)
        } else if (msg.type === 'complete') {
          progress.pct = 100
          api.getRun(runId).then(r => { run = r })
        }
      })
    }
  })

  onDestroy(() => liveWs?.close())
</script>

<h2>Run Detail — <code>{runId?.slice(0,8)}…</code></h2>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

{#if run?.status === 'running'}
  <div style="margin:1rem 0;">
    <div style="background:#313244;border-radius:4px;height:20px;width:100%;">
      <div style="background:#89b4fa;height:100%;width:{progress.pct}%;border-radius:4px;transition:width 0.3s;"></div>
    </div>
    <small>{progress.done}/{progress.total} segments ({progress.pct}%) — ETA {progress.eta_s}s</small>
  </div>
{/if}

{#if run?.aggregate}
  <div style="display:flex;gap:2rem;margin:1rem 0;padding:1rem;background:#313244;border-radius:8px;">
    <span>WER: <strong>{run.aggregate.wer_mean?.toFixed(3)}</strong></span>
    <span>CER: <strong>{run.aggregate.cer_mean?.toFixed(3)}</strong></span>
    <span>RTFx: <strong>{run.aggregate.rtfx_mean?.toFixed(1)}×</strong></span>
    <span>VRAM: <strong>{run.aggregate.vram_peak_mb?.toFixed(0)} MB</strong></span>
    <span>Time: <strong>{run.aggregate.wall_time_s?.toFixed(1)}s</strong></span>
  </div>
{/if}

<h3>Segments</h3>
<table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
  <thead>
    <tr style="color:#89b4fa;text-align:left;">
      <th>Offset</th><th>Duration</th><th>WER</th><th>RTFx</th><th>Reference</th><th>Hypothesis</th>
    </tr>
  </thead>
  <tbody>
    {#each segments as seg}
      <tr style="border-top:1px solid #313244;">
        <td style="padding:0.3rem 0;">{seg.offset_s?.toFixed(1)}s</td>
        <td>{seg.duration_s?.toFixed(1)}s</td>
        <td style="color:{seg.wer > 0.2 ? '#f38ba8' : '#a6e3a1'}">{seg.wer?.toFixed(3) ?? '—'}</td>
        <td>{seg.rtfx?.toFixed(1) ?? '—'}</td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title={seg.ref_text}>{seg.ref_text}</td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title={seg.hyp_text}>{seg.hyp_text}</td>
      </tr>
    {/each}
  </tbody>
</table>

<div style="margin-top:1rem;display:flex;gap:0.5rem;">
  <a href={api.exportRun(runId,'json')} download>Export JSON</a>
  <a href={api.exportRun(runId,'csv')} download>Export CSV</a>
</div>
