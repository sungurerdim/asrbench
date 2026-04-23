<script lang="ts">
  import { onMount } from 'svelte'
  import { connectOptimize } from '../lib/ws'

  type Study = {
    study_id: string
    status?: string
    total_trials?: number
    best_score?: number | null
    phases?: Record<string, number>
  }

  let studies = $state<Study[]>([])
  let liveUpdates = $state<Record<string, Study>>({})
  let sockets = $state<Record<string, WebSocket>>({})
  let error = $state<string>('')

  async function fetchStudies(): Promise<void> {
    try {
      const res = await fetch('/optimize/')
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      studies = (await res.json()) as Study[]
      // Subscribe to every non-terminal study so the UI updates without a refresh.
      for (const s of studies) {
        if (sockets[s.study_id]) continue
        if (s.status === 'completed' || s.status === 'failed') continue
        sockets[s.study_id] = connectOptimize(s.study_id, (msg) => {
          liveUpdates = { ...liveUpdates, [s.study_id]: msg as Study }
        })
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to list studies'
    }
  }

  onMount(() => {
    void fetchStudies()
    return () => {
      for (const ws of Object.values(sockets)) {
        ws.close()
      }
    }
  })

  function statusColor(s: string | undefined): string {
    if (s === 'completed') return '#a6e3a1'
    if (s === 'failed') return '#f38ba8'
    if (s === 'running') return '#89b4fa'
    return '#cdd6f4'
  }

  function displayOf(s: Study): Study {
    return { ...s, ...(liveUpdates[s.study_id] ?? {}) }
  }
</script>

<h2>Optimization Studies</h2>
<p style="color:#a6adc8;font-size:0.85rem;">
  Every IAMS study goes through 7 search layers. Live updates arrive over a
  WebSocket subscription, so trial counts and the current best score update
  without refreshing.
</p>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

{#if studies.length === 0}
  <p style="color:#a6adc8;">
    No studies yet. Start one via <code>POST /optimize/start</code> or the
    <code>asrbench optimize start</code> CLI command.
  </p>
{:else}
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
    <thead>
      <tr style="color:#89b4fa;text-align:left;">
        <th>Study ID</th><th>Status</th><th>Trials</th><th>Best score</th><th>Phases</th>
      </tr>
    </thead>
    <tbody>
      {#each studies as raw (raw.study_id)}
        {@const s = displayOf(raw)}
        <tr style="border-top:1px solid #313244;">
          <td style="font-family:monospace;">{s.study_id.slice(0, 8)}…</td>
          <td style="color:{statusColor(s.status)}">{s.status ?? '—'}</td>
          <td>{s.total_trials ?? 0}</td>
          <td>{s.best_score != null ? Number(s.best_score).toFixed(4) : '—'}</td>
          <td>
            {#if s.phases}
              {Object.entries(s.phases)
                .map(([p, n]) => `${p}:${n}`)
                .join(' · ')}
            {:else}
              —
            {/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}
