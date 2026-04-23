<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../lib/api'
  import type { CompareResponse, Run } from '../lib/types'

  let runs = $state<Run[]>([])
  let selected = $state<Set<string>>(new Set())
  let result = $state<CompareResponse | null>(null)
  let error = $state<string>('')
  let loading = $state<boolean>(false)

  onMount(async () => {
    try {
      runs = await api.listRuns({ status: 'completed', limit: 50 })
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to list runs'
    }
  })

  function toggle(runId: string): void {
    const next = new Set(selected)
    if (next.has(runId)) {
      next.delete(runId)
    } else {
      next.add(runId)
    }
    selected = next
  }

  async function runCompare(): Promise<void> {
    if (selected.size < 2) {
      error = 'Select at least 2 completed runs.'
      return
    }
    error = ''
    loading = true
    try {
      result = await api.compareRuns(Array.from(selected))
    } catch (e) {
      error = e instanceof Error ? e.message : 'Compare failed'
      result = null
    } finally {
      loading = false
    }
  }

  function fmt(value: unknown, digits = 4): string {
    if (typeof value !== 'number') return '—'
    return value.toFixed(digits)
  }

  function fmtDelta(value: unknown, digits = 4): string {
    if (typeof value !== 'number') return ''
    const sign = value > 0 ? '+' : ''
    return `${sign}${value.toFixed(digits)}`
  }
</script>

<h2>Compare Runs</h2>
<p style="color:#a6adc8;font-size:0.85rem;">
  Select two or more completed runs to see shared vs. differing parameters and
  per-run deltas vs. the baseline (first row).
</p>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

<section style="display:flex;gap:2rem;align-items:flex-start;margin-top:1rem;">
  <div style="flex:1;max-width:520px;">
    <h3>Select runs</h3>
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
      <thead>
        <tr style="color:#89b4fa;text-align:left;">
          <th></th><th>ID</th><th>Backend</th><th>Lang</th><th>WER</th>
        </tr>
      </thead>
      <tbody>
        {#each runs as run (run.run_id)}
          <tr style="border-top:1px solid #313244;">
            <td>
              <input
                type="checkbox"
                checked={selected.has(run.run_id)}
                onchange={() => toggle(run.run_id)}
              />
            </td>
            <td style="font-family:monospace;">{run.run_id.slice(0, 8)}…</td>
            <td>{run.backend}</td>
            <td>{run.lang}</td>
            <td>{fmt(run.aggregate?.wer_mean ?? null)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <button
      onclick={runCompare}
      disabled={loading || selected.size < 2}
      style="margin-top:0.75rem;padding:0.5rem 1rem;background:#89b4fa;color:#1e1e2e;border:none;border-radius:4px;font-weight:bold;"
    >
      {loading ? 'Comparing…' : `Compare ${selected.size} run${selected.size === 1 ? '' : 's'}`}
    </button>
  </div>

  <div style="flex:1;">
    {#if result}
      <h3>Results</h3>
      <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
        <thead>
          <tr style="color:#89b4fa;text-align:left;">
            <th>Run</th><th>WER</th><th>Δ WER</th><th>RTFx</th><th>Δ RTFx</th>
          </tr>
        </thead>
        <tbody>
          {#each result.runs as row (row.run_id)}
            <tr style="border-top:1px solid #313244;">
              <td style="font-family:monospace;">
                {row.run_id.slice(0, 8)}…{row.is_baseline ? ' ⭐' : ''}
              </td>
              <td>{fmt(row.aggregate.wer_mean)}</td>
              <td style={Number(row.delta_wer_mean) < 0 ? 'color:#a6e3a1' : 'color:#f38ba8'}>
                {fmtDelta(row.delta_wer_mean)}
              </td>
              <td>{fmt(row.aggregate.rtfx_mean, 2)}</td>
              <td style={Number(row.delta_rtfx_mean) > 0 ? 'color:#a6e3a1' : 'color:#f38ba8'}>
                {fmtDelta(row.delta_rtfx_mean, 2)}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>

      {#if result.params_same.length}
        <h4 style="margin-top:1.5rem;">Shared params</h4>
        <ul>
          {#each result.params_same as key (key)}
            <li>
              <code>{key}</code> = {String(result.runs[0].params[key])}
            </li>
          {/each}
        </ul>
      {/if}

      {#if result.params_diff.length}
        <h4 style="margin-top:1rem;">Differing params</h4>
        <ul>
          {#each result.params_diff as key (key)}
            <li>
              <code>{key}</code>:
              {result.runs.map((r) => String(r.params[key])).join(' | ')}
            </li>
          {/each}
        </ul>
      {/if}

      {#if result.wilcoxon_p !== null && result.wilcoxon_p !== undefined}
        <p style="margin-top:1rem;">
          Wilcoxon signed-rank
          <strong>p = {result.wilcoxon_p.toFixed(4)}</strong> —
          {result.wilcoxon_p < 0.05 ? 'significant at p<0.05' : 'not significant'}
        </p>
      {/if}
    {:else}
      <p style="color:#a6adc8;">Pick runs on the left, then hit Compare.</p>
    {/if}
  </div>
</section>
