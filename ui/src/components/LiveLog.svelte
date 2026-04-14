<script>
  let { logs = [] } = $props()

  function levelColor(l) {
    return { error: '#f38ba8', warn: '#fab387', info: '#89b4fa', debug: '#6c7086' }[l] || '#cdd6f4'
  }
</script>

<div style="background:#181825;border-radius:6px;padding:0.5rem;height:200px;overflow-y:auto;font-size:0.75rem;font-family:monospace;">
  {#each logs as entry}
    <div>
      <span style="color:#6c7086;">{new Date(entry.ts * 1000).toLocaleTimeString()}</span>
      <span style="color:{levelColor(entry.level)};margin:0 0.4rem;">[{entry.level?.toUpperCase()}]</span>
      <span>{entry.msg}</span>
    </div>
  {/each}
  {#if logs.length === 0}
    <span style="color:#6c7086;">No activity yet…</span>
  {/if}
</div>
