<script lang="ts">
  import type { ActivityRecord } from '../lib/types'

  interface Props {
    logs?: ActivityRecord[]
  }

  let { logs = [] }: Props = $props()

  function levelColor(l: string): string {
    const map: Record<string, string> = {
      error: '#f38ba8',
      warning: '#fab387',
      warn: '#fab387',
      info: '#89b4fa',
      debug: '#6c7086',
    }
    return map[l] ?? '#cdd6f4'
  }

  function formatTs(ts: unknown): string {
    if (typeof ts === 'string' && ts.length > 0) {
      // ISO8601 from the activity logger — trim to HH:MM:SS for display.
      const d = new Date(ts)
      if (!Number.isNaN(d.getTime())) return d.toLocaleTimeString()
      return ts.slice(0, 19)
    }
    if (typeof ts === 'number') return new Date(ts * 1000).toLocaleTimeString()
    return ''
  }
</script>

<div
  style="background:#181825;border-radius:6px;padding:0.5rem;height:200px;overflow-y:auto;font-size:0.75rem;font-family:monospace;"
>
  {#each logs as entry, i (i)}
    <div>
      <span style="color:#6c7086;">{formatTs(entry.ts)}</span>
      <span style="color:{levelColor(entry.level)};margin:0 0.4rem;">
        [{entry.level?.toUpperCase()}]
      </span>
      <span>{entry.message}</span>
    </div>
  {/each}
  {#if logs.length === 0}
    <span style="color:#6c7086;">No activity yet…</span>
  {/if}
</div>
