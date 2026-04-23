<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../lib/api'

  let health = $state<{ status: string; version: string } | null>(null)
  let vram = $state<{ gpu_available: boolean; gpus: Array<Record<string, unknown>> } | null>(null)
  let error = $state<string>('')

  onMount(async () => {
    try {
      health = await api.health()
      vram = await api.vram()
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to query server'
    }
  })
</script>

<h2>Doctor</h2>
<p style="color:#a6adc8;font-size:0.85rem;">
  Environment overview as reported by the running ASRbench server. For a deeper
  check (ffmpeg, backends, Turkish normalizer, disk space) run
  <code>asrbench doctor</code> from your terminal.
</p>

{#if error}
  <p style="color:#f38ba8">{error}</p>
{/if}

<h3>Server</h3>
{#if health}
  <ul>
    <li>Status: <span style="color:#a6e3a1">{health.status}</span></li>
    <li>Version: <code>{health.version}</code></li>
  </ul>
{:else}
  <p>Loading…</p>
{/if}

<h3>GPU / VRAM</h3>
{#if vram}
  {#if vram.gpu_available}
    <ul>
      {#each vram.gpus as gpu (gpu.index)}
        <li>
          <strong>{gpu.name}</strong> — {Number(gpu.vram_used_mb).toFixed(0)} MB used of
          {Number(gpu.vram_total_mb).toFixed(0)} MB total
        </li>
      {/each}
    </ul>
  {:else}
    <p style="color:#fab387">
      No NVIDIA GPU visible — CPU-only backends (whisper.cpp) still work.
    </p>
  {/if}
{:else}
  <p>Loading…</p>
{/if}
