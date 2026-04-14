<script>
  import { api } from '../lib/api.js'

  let runIdsInput = $state('')
  let result = $state(null)
  let error = $state('')
  let loading = $state(false)

  async function compare() {
    loading = true
    error = ''
    try {
      const ids = runIdsInput.split(',').map(s => s.trim()).filter(Boolean)
      result = await api.compareRuns(ids)
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }
</script>

<h2>Compare Runs</h2>

<div style="display:flex;gap:0.5rem;margin-bottom:1rem;">
  <input bind:value={runIdsInput} placeholder="uuid1, uuid2, uuid3" style="flex:1;padding:0.5rem;background:#313244;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;" />
  <button onclick={compare} disabled={loading} style="padding:0.5rem 1rem;background:#89b4fa;color:#1e1e2e;border:none;border-radius:4px;">
    {loading ? 'Loading…' : 'Compare'}
  </button>
</div>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

{#if result}
  <p style="color:#6c7086;">Params differ: <strong>{result.params_diff.join(', ') || '(none)'}</strong> | Same: {result.params_same.join(', ')}</p>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
    <thead>
      <tr style="color:#89b4fa;text-align:left;">
        <th>Run</th><th>WER</th><th>ΔWER</th><th>CER</th><th>RTFx</th><th>ΔRTFx</th><th>VRAM</th><th>Baseline</th>
      </tr>
    </thead>
    <tbody>
      {#each result.runs as r}
        <tr style="border-top:1px solid #313244;{r.is_baseline?'background:#313244':''}">
          <td style="padding:0.4rem 0;">{r.run_id.slice(0,8)}…</td>
          <td>{r.wer_mean?.toFixed(3) ?? '—'}</td>
          <td style="color:{r.delta_wer < 0 ? '#a6e3a1' : r.delta_wer > 0 ? '#f38ba8' : '#cdd6f4'}">
            {r.delta_wer != null ? (r.delta_wer > 0 ? '+' : '') + r.delta_wer.toFixed(3) : '—'}
          </td>
          <td>{r.cer_mean?.toFixed(3) ?? '—'}</td>
          <td>{r.rtfx_mean?.toFixed(1) ?? '—'}</td>
          <td style="color:{r.delta_rtfx < 0 ? '#a6e3a1' : r.delta_rtfx > 0 ? '#f38ba8' : '#cdd6f4'}">
            {r.delta_rtfx != null ? (r.delta_rtfx > 0 ? '+' : '') + r.delta_rtfx.toFixed(1) : '—'}
          </td>
          <td>{r.vram_peak_mb?.toFixed(0) ?? '—'} MB</td>
          <td>{r.is_baseline ? '✓' : ''}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}
