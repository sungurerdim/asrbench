<script>
  import { onMount, onDestroy } from 'svelte'
  import { api } from '../lib/api.js'
  import { connectVRAM, connectLogs } from '../lib/ws.js'
  import VRAMBar from '../components/VRAMBar.svelte'
  import LiveLog from '../components/LiveLog.svelte'

  let { onRunSelect } = $props()

  let runs = $state([])
  let vram = $state({ used_mb: 0, total_mb: 0, pct: 0, available: false })
  let logs = $state([])
  let error = $state('')

  let vramWs, logsWs

  onMount(async () => {
    try {
      runs = await api.listRuns({ limit: 50 })
    } catch (e) {
      error = e.message
    }

    vramWs = connectVRAM((snap) => { vram = snap })
    logsWs = connectLogs((entry) => {
      logs = [entry, ...logs].slice(0, 100)
    })
  })

  onDestroy(() => {
    vramWs?.close()
    logsWs?.close()
  })

  function statusColor(s) {
    return { completed: '#a6e3a1', failed: '#f38ba8', running: '#89b4fa', cancelled: '#fab387', pending: '#cdd6f4' }[s] || '#cdd6f4'
  }
</script>

<h2>Dashboard</h2>

<VRAMBar {vram} />

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

<h3>Recent Runs</h3>
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
  <thead>
    <tr style="color:#89b4fa;text-align:left;">
      <th>ID</th><th>Backend</th><th>Lang</th><th>Status</th><th>WER</th><th>RTFx</th><th>Created</th>
    </tr>
  </thead>
  <tbody>
    {#each runs as run}
      <tr style="border-top:1px solid #313244;cursor:pointer;" onclick={() => onRunSelect(run.run_id)}>
        <td style="padding:0.4rem 0;">{run.run_id.slice(0,8)}…</td>
        <td>{run.backend}</td>
        <td>{run.lang}</td>
        <td style="color:{statusColor(run.status)}">{run.status}</td>
        <td>{run.aggregate?.wer_mean?.toFixed(3) ?? '—'}</td>
        <td>{run.aggregate?.rtfx_mean?.toFixed(1) ?? '—'}</td>
        <td>{run.created_at?.slice(0,19)}</td>
      </tr>
    {/each}
  </tbody>
</table>

<h3 style="margin-top:2rem;">Activity Log</h3>
<LiveLog {logs} />
