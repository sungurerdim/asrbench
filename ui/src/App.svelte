<script>
  import { onMount } from 'svelte'
  import Dashboard from './routes/Dashboard.svelte'
  import RunDetail from './routes/RunDetail.svelte'
  import Compare from './routes/Compare.svelte'
  import Models from './routes/Models.svelte'
  import Datasets from './routes/Datasets.svelte'

  let page = $state(location.hash.replace('#', '') || 'dashboard')
  let runDetailId = $state(null)

  function navigate(to, params = {}) {
    page = to
    if (params.runId) runDetailId = params.runId
    location.hash = to
  }

  onMount(() => {
    window.addEventListener('hashchange', () => {
      const hash = location.hash.replace('#', '') || 'dashboard'
      if (!hash.startsWith('run/')) page = hash
    })
  })
</script>

<nav style="display:flex;gap:1rem;padding:0.75rem 1rem;background:#1e1e2e;color:#cdd6f4;font-family:monospace;">
  <strong style="margin-right:1rem;color:#89b4fa;">asrbench</strong>
  {#each [['dashboard','Dashboard'],['models','Models'],['datasets','Datasets']] as [id,label]}
    <button onclick={() => navigate(id)} style="background:none;border:none;color:{page===id?'#89b4fa':'#cdd6f4'};cursor:pointer;font-size:0.9rem;">{label}</button>
  {/each}
</nav>

<main style="padding:1rem;font-family:monospace;background:#1e1e2e;min-height:100vh;color:#cdd6f4;">
  {#if page === 'dashboard'}
    <Dashboard onRunSelect={(id) => navigate('run-detail', { runId: id })} />
  {:else if page === 'run-detail' && runDetailId}
    <RunDetail runId={runDetailId} />
  {:else if page === 'compare'}
    <Compare />
  {:else if page === 'models'}
    <Models />
  {:else if page === 'datasets'}
    <Datasets />
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #1e1e2e; }
  :global(button) { cursor: pointer; }
</style>
