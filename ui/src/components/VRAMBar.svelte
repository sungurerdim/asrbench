<script lang="ts">
  import type { VRAMSnapshot } from '../lib/types'

  interface Props {
    vram: VRAMSnapshot
  }

  let { vram }: Props = $props()
</script>

<div style="margin:0.5rem 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.25rem;">
    <span>VRAM</span>
    {#if vram.available}
      <span>
        {vram.used_mb.toFixed(0)} / {vram.total_mb.toFixed(0)} MB ({vram.pct.toFixed(1)}%)
      </span>
    {:else}
      <span style="color:#6c7086;">Not available</span>
    {/if}
  </div>
  {#if vram.available}
    <div style="background:#313244;border-radius:4px;height:8px;width:100%;">
      <div
        style="background:{vram.pct > 85
          ? '#f38ba8'
          : vram.pct > 60
            ? '#fab387'
            : '#a6e3a1'};height:100%;width:{Math.min(
          vram.pct,
          100,
        )}%;border-radius:4px;transition:width 0.5s;"
      ></div>
    </div>
  {/if}
</div>
