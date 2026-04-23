<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { api } from '../lib/api'
  import { connectVRAM, connectActivity } from '../lib/ws'
  import type { ActivityRecord, Run, VRAMSnapshot } from '../lib/types'
  import VRAMBar from '../components/VRAMBar.svelte'
  import LiveLog from '../components/LiveLog.svelte'

  interface Props {
    onRunSelect: (id: string) => void
  }

  let { onRunSelect }: Props = $props()

  let runs = $state<Run[]>([])
  let vram = $state<VRAMSnapshot>({
    available: false,
    used_mb: 0,
    total_mb: 0,
    pct: 0,
  })
  let logs = $state<ActivityRecord[]>([])
  let error = $state<string>('')

  let vramWs: WebSocket | null = null
  let activityWs: WebSocket | null = null

  let totalRuns = $derived(runs.length)
  let completedRuns = $derived(runs.filter((r) => r.status === 'completed'))
  let avgWer = $derived(
    completedRuns.length > 0
      ? completedRuns
          .map((r) => r.aggregate?.wer_mean ?? null)
          .filter((v): v is number => typeof v === 'number')
          .reduce((a, b, _, arr) => a + b / arr.length, 0)
      : null,
  )
  let bestRun = $derived(
    completedRuns
      .filter((r) => typeof r.aggregate?.wer_mean === 'number')
      .sort((a, b) => (a.aggregate!.wer_mean ?? 1) - (b.aggregate!.wer_mean ?? 1))[0],
  )

  onMount(async () => {
    try {
      runs = await api.listRuns({ limit: 50 })
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load runs'
    }

    vramWs = connectVRAM((snap) => {
      vram = snap
    })
    activityWs = connectActivity((entry) => {
      logs = [entry, ...logs].slice(0, 100)
    })
  })

  onDestroy(() => {
    vramWs?.close()
    activityWs?.close()
  })

  function statusColor(s: string): string {
    const map: Record<string, string> = {
      completed: '#a6e3a1',
      failed: '#f38ba8',
      running: '#89b4fa',
      cancelled: '#fab387',
      pending: '#cdd6f4',
    }
    return map[s] ?? '#cdd6f4'
  }
</script>

<h2>Dashboard</h2>

<section
  style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:0.75rem;margin:0.75rem 0 1.25rem;"
>
  <div style="padding:0.75rem;background:#313244;border-radius:6px;">
    <div style="font-size:0.7rem;color:#a6adc8;">Total runs</div>
    <div style="font-size:1.5rem;font-weight:bold;">{totalRuns}</div>
  </div>
  <div style="padding:0.75rem;background:#313244;border-radius:6px;">
    <div style="font-size:0.7rem;color:#a6adc8;">Completed</div>
    <div style="font-size:1.5rem;font-weight:bold;">{completedRuns.length}</div>
  </div>
  <div style="padding:0.75rem;background:#313244;border-radius:6px;">
    <div style="font-size:0.7rem;color:#a6adc8;">Avg WER (completed)</div>
    <div style="font-size:1.5rem;font-weight:bold;">
      {avgWer !== null ? avgWer.toFixed(4) : '—'}
    </div>
  </div>
  <div style="padding:0.75rem;background:#313244;border-radius:6px;">
    <div style="font-size:0.7rem;color:#a6adc8;">Best run</div>
    <div style="font-size:0.9rem;font-family:monospace;">
      {bestRun ? bestRun.run_id.slice(0, 8) + '…' : '—'}
    </div>
    <div style="font-size:0.75rem;color:#a6adc8;">
      {bestRun?.aggregate?.wer_mean !== undefined && bestRun.aggregate.wer_mean !== null
        ? `WER ${bestRun.aggregate.wer_mean.toFixed(4)}`
        : ''}
    </div>
  </div>
</section>

<VRAMBar {vram} />

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

<h3>Recent Runs</h3>
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
  <thead>
    <tr style="color:#89b4fa;text-align:left;">
      <th>ID</th><th>Backend</th><th>Lang</th><th>Status</th><th>WER</th><th>RTFx</th>
    </tr>
  </thead>
  <tbody>
    {#each runs as run (run.run_id)}
      <tr
        style="border-top:1px solid #313244;cursor:pointer;"
        onclick={() => onRunSelect(run.run_id)}
      >
        <td style="padding:0.4rem 0;">{run.run_id.slice(0, 8)}…</td>
        <td>{run.backend}</td>
        <td>{run.lang}</td>
        <td style="color:{statusColor(run.status)}">{run.status}</td>
        <td>{run.aggregate?.wer_mean?.toFixed(3) ?? '—'}</td>
        <td>{run.aggregate?.rtfx_mean?.toFixed(1) ?? '—'}</td>
      </tr>
    {/each}
  </tbody>
</table>

<h3 style="margin-top:2rem;">Activity Log</h3>
<LiveLog {logs} />
