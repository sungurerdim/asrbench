<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../lib/api'
  import type { Dataset, Model } from '../lib/types'

  interface Props {
    onDone?: () => void
  }

  let { onDone }: Props = $props()

  let step = $state<1 | 2 | 3 | 4>(1)
  let models = $state<Model[]>([])
  let datasets = $state<Dataset[]>([])
  let selectedModel = $state<string>('')
  let selectedDataset = $state<string>('')
  let lang = $state<string>('en')
  let error = $state<string>('')
  let starting = $state<boolean>(false)

  onMount(async () => {
    try {
      models = await api.listModels()
      datasets = await api.listDatasets()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to inspect server'
    }
  })

  function next(): void {
    if (step < 4) step = ((step + 1) as 1 | 2 | 3 | 4)
  }

  function back(): void {
    if (step > 1) step = ((step - 1) as 1 | 2 | 3 | 4)
  }

  async function startRun(): Promise<void> {
    if (!selectedModel || !selectedDataset) {
      error = 'Pick a model and a dataset first.'
      return
    }
    starting = true
    error = ''
    try {
      await api.startRun({
        model_id: selectedModel,
        dataset_id: selectedDataset,
        lang,
        mode: 'model_compare',
      })
      onDone?.()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Start run failed'
    } finally {
      starting = false
    }
  }
</script>

<h2>First Benchmark</h2>
<p style="color:#a6adc8;font-size:0.85rem;">Four short steps and you are running a benchmark.</p>

<nav style="display:flex;gap:0.5rem;margin-bottom:1rem;">
  {#each [1, 2, 3, 4] as n (n)}
    <span
      style="padding:0.25rem 0.75rem;border-radius:4px;background:{step === n
        ? '#89b4fa'
        : '#313244'};color:{step === n ? '#1e1e2e' : '#cdd6f4'};font-weight:bold;"
    >
      {n}
    </span>
  {/each}
</nav>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

<section style="border:1px solid #313244;padding:1rem;border-radius:6px;">
  {#if step === 1}
    <h3>1 — Pick a model</h3>
    {#if models.length === 0}
      <p style="color:#fab387;">
        No models registered yet. Register one with <code>asrbench models register</code>
        or the Models page, then come back here.
      </p>
    {:else}
      <ul style="list-style:none;padding:0;">
        {#each models as m (m.model_id)}
          <li style="padding:0.4rem 0;">
            <label>
              <input type="radio" bind:group={selectedModel} value={m.model_id} />
              <strong>{m.name}</strong> <span style="color:#a6adc8">({m.backend})</span>
            </label>
          </li>
        {/each}
      </ul>
    {/if}
  {:else if step === 2}
    <h3>2 — Pick a dataset</h3>
    {#if datasets.length === 0}
      <p style="color:#fab387;">
        No datasets yet. Fetch one from the Datasets page — FLEURS tr / en is a
        good minimal start (~30 MB download).
      </p>
    {:else}
      <ul style="list-style:none;padding:0;">
        {#each datasets as d (d.dataset_id)}
          <li style="padding:0.4rem 0;">
            <label>
              <input type="radio" bind:group={selectedDataset} value={d.dataset_id} />
              <strong>{d.name}</strong>
              <span style="color:#a6adc8;">
                ({d.source} · {d.lang} · {d.split}{d.verified ? '' : ' · unverified'})
              </span>
            </label>
          </li>
        {/each}
      </ul>
    {/if}
  {:else if step === 3}
    <h3>3 — Language</h3>
    <p style="color:#a6adc8;font-size:0.85rem;">
      ISO 639-1 — controls WER normalization (TR is agglutinative, CER is the
      preferred metric there; tooltip on the Dashboard).
    </p>
    <select bind:value={lang} style="font-size:1rem;padding:0.25rem;">
      {#each ['en', 'tr', 'ar', 'zh', 'ja', 'ko', 'de', 'es', 'fr'] as code (code)}
        <option value={code}>{code}</option>
      {/each}
    </select>
  {:else if step === 4}
    <h3>4 — Review and start</h3>
    <ul>
      <li>Model: <code>{selectedModel || '—'}</code></li>
      <li>Dataset: <code>{selectedDataset || '—'}</code></li>
      <li>Language: <code>{lang}</code></li>
    </ul>
    <button
      onclick={startRun}
      disabled={starting || !selectedModel || !selectedDataset}
      style="padding:0.5rem 1rem;background:#89b4fa;color:#1e1e2e;border:none;border-radius:4px;font-weight:bold;"
    >
      {starting ? 'Starting…' : 'Start benchmark'}
    </button>
  {/if}
</section>

<div style="margin-top:1rem;display:flex;gap:0.5rem;">
  <button onclick={back} disabled={step === 1}>Back</button>
  {#if step < 4}
    <button onclick={next}>Next</button>
  {/if}
</div>
