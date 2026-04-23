<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../lib/api'
  import type { Model } from '../lib/types'

  let models = $state<Model[]>([])
  let error = $state<string>('')
  let loading = $state<boolean>(false)

  let form = $state<{ family: string; name: string; backend: string; local_path: string }>({
    family: '',
    name: '',
    backend: 'faster-whisper',
    local_path: '',
  })

  onMount(() => load())

  async function load(): Promise<void> {
    try {
      models = await api.listModels()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to list models'
    }
  }

  async function register(): Promise<void> {
    loading = true
    try {
      await api.registerModel(form)
      form = { family: '', name: '', backend: 'faster-whisper', local_path: '' }
      await load()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Register failed'
    } finally {
      loading = false
    }
  }

  async function loadModel(id: string): Promise<void> {
    try {
      await api.loadModel(id)
      await load()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Load failed'
    }
  }

  async function unloadModel(id: string): Promise<void> {
    try {
      await api.unloadModel(id)
      await load()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unload failed'
    }
  }
</script>

<h2>Models</h2>

{#if error}<p style="color:#f38ba8">{error}</p>{/if}

<details style="margin-bottom:1rem;">
  <summary style="cursor:pointer;color:#89b4fa;">Register New Model</summary>
  <div
    style="padding:1rem;background:#313244;border-radius:8px;margin-top:0.5rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;"
  >
    <input
      bind:value={form.family}
      placeholder="Family (e.g. whisper)"
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    />
    <input
      bind:value={form.name}
      placeholder="Name (e.g. large-v3)"
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    />
    <input
      bind:value={form.backend}
      placeholder="Backend"
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    />
    <input
      bind:value={form.local_path}
      placeholder="/path/to/model"
      style="padding:0.4rem;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;border-radius:4px;"
    />
    <button
      onclick={register}
      disabled={loading}
      style="grid-column:span 2;padding:0.5rem;background:#89b4fa;color:#1e1e2e;border:none;border-radius:4px;"
    >
      {loading ? 'Registering…' : 'Register'}
    </button>
  </div>
</details>

<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
  <thead>
    <tr style="color:#89b4fa;text-align:left;">
      <th>ID</th><th>Name</th><th>Backend</th><th>Path</th><th>Actions</th>
    </tr>
  </thead>
  <tbody>
    {#each models as m (m.model_id)}
      <tr style="border-top:1px solid #313244;">
        <td style="padding:0.4rem 0;">{m.model_id.slice(0, 8)}…</td>
        <td>{m.name}</td>
        <td>{m.backend}</td>
        <td
          style="font-size:0.75rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;"
          title={m.local_path}
        >
          {m.local_path}
        </td>
        <td style="display:flex;gap:0.4rem;">
          <button
            onclick={() => loadModel(m.model_id)}
            style="padding:0.2rem 0.5rem;background:#89b4fa;color:#1e1e2e;border:none;border-radius:3px;font-size:0.75rem;"
          >
            {m.loaded ? 'Reload' : 'Load'}
          </button>
          <button
            onclick={() => unloadModel(m.model_id)}
            style="padding:0.2rem 0.5rem;background:#45475a;color:#cdd6f4;border:none;border-radius:3px;font-size:0.75rem;"
          >
            Unload
          </button>
        </td>
      </tr>
    {/each}
  </tbody>
</table>
