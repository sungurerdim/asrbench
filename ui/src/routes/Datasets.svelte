<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../lib/api'
  import type { Dataset } from '../lib/types'

  let datasets = $state<Dataset[]>([])
  let error = $state<string>('')
  let loading = $state<boolean>(false)
  let form = $state<{ source: string; lang: string; split: string; local_path: string }>({
    source: 'common_voice',
    lang: 'en',
    split: 'test',
    local_path: '',
  })

  onMount(() => load())

  async function load(): Promise<void> {
    try {
      datasets = await api.listDatasets()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to list datasets'
    }
  }

  async function fetchDataset(): Promise<void> {
    loading = true
    try {
      await api.fetchDataset(form)
      error = ''
      setTimeout(load, 2000)
    } catch (e) {
      error = e instanceof Error ? e.message : 'Fetch failed'
    } finally {
      loading = false
    }
  }

  async function deleteDataset(id: string): Promise<void> {
    if (!confirm('Delete dataset?')) return
    try {
      await api.deleteDataset(id)
      await load()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Delete failed'
    }
  }
</script>

<h2>Datasets</h2>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

<details style="margin-bottom:1rem;">
  <summary style="cursor:pointer;color:#89b4fa;">Download New Dataset</summary>
  <div
    style="padding:1rem;background:#313244;border-radius:8px;margin-top:0.5rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;"
  >
    <select
      bind:value={form.source}
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    >
      {#each ['common_voice', 'fleurs', 'yodas', 'ted_lium', 'librispeech', 'earnings22', 'mediaspeech', 'custom'] as s (s)}
        <option value={s}>{s}</option>
      {/each}
    </select>
    <input
      bind:value={form.lang}
      placeholder="Language (e.g. en)"
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    />
    <select
      bind:value={form.split}
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    >
      {#each ['test', 'validation', 'train'] as s (s)}
        <option value={s}>{s}</option>
      {/each}
    </select>
    {#if form.source === 'custom'}
      <input
        bind:value={form.local_path}
        placeholder="/path/to/audio"
        style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
      />
    {:else}
      <div></div>
    {/if}
    <button
      onclick={fetchDataset}
      disabled={loading}
      style="grid-column:span 2;padding:0.5rem;background:#89b4fa;color:#1e1e2e;border:none;border-radius:4px;"
    >
      {loading ? 'Starting download…' : 'Download'}
    </button>
  </div>
</details>

<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
  <thead>
    <tr style="color:#89b4fa;text-align:left;">
      <th>ID</th><th>Name</th><th>Lang</th><th>Split</th><th>Duration</th><th>Verified</th
      ><th>Actions</th>
    </tr>
  </thead>
  <tbody>
    {#each datasets as d (d.dataset_id)}
      <tr style="border-top:1px solid #313244;">
        <td style="padding:0.4rem 0;">{d.dataset_id.slice(0, 8)}…</td>
        <td>{d.name}</td>
        <td>{d.lang}</td>
        <td>{d.split}</td>
        <td>{d.duration_s ? d.duration_s.toFixed(0) + 's' : '—'}</td>
        <td style="color:{d.verified ? '#a6e3a1' : '#fab387'}">
          {d.verified ? '✓' : '…'}
        </td>
        <td>
          <button
            onclick={() => deleteDataset(d.dataset_id)}
            style="padding:0.2rem 0.5rem;background:#f38ba8;color:#1e1e2e;border:none;border-radius:3px;font-size:0.75rem;"
          >
            Delete
          </button>
        </td>
      </tr>
    {/each}
  </tbody>
</table>
