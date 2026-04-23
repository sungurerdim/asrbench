<script lang="ts">
  import { onMount } from 'svelte'
  import Dashboard from './routes/Dashboard.svelte'
  import RunDetail from './routes/RunDetail.svelte'
  import Compare from './routes/Compare.svelte'
  import Models from './routes/Models.svelte'
  import Datasets from './routes/Datasets.svelte'
  import Optimize from './routes/Optimize.svelte'
  import Doctor from './routes/Doctor.svelte'
  import Onboarding from './routes/Onboarding.svelte'

  type Page =
    | 'dashboard'
    | 'run-detail'
    | 'compare'
    | 'models'
    | 'datasets'
    | 'optimize'
    | 'doctor'
    | 'onboarding'

  let page = $state<Page>((location.hash.replace('#', '') as Page) || 'dashboard')
  let runDetailId = $state<string | null>(null)

  function navigate(to: Page, params: { runId?: string } = {}): void {
    page = to
    if (params.runId) runDetailId = params.runId
    location.hash = to
  }

  onMount(() => {
    window.addEventListener('hashchange', () => {
      const hash = (location.hash.replace('#', '') || 'dashboard') as Page
      if (!hash.startsWith('run/')) page = hash
    })
  })

  const navItems: Array<[Page, string]> = [
    ['dashboard', 'Dashboard'],
    ['compare', 'Compare'],
    ['optimize', 'Optimize'],
    ['models', 'Models'],
    ['datasets', 'Datasets'],
    ['doctor', 'Doctor'],
  ]
</script>

<nav
  style="display:flex;gap:1rem;padding:0.75rem 1rem;background:#1e1e2e;color:#cdd6f4;font-family:monospace;border-bottom:1px solid #313244;"
>
  <strong style="margin-right:1rem;color:#89b4fa;">asrbench</strong>
  {#each navItems as [id, label]}
    <button
      onclick={() => navigate(id)}
      style="background:none;border:none;color:{page === id
        ? '#89b4fa'
        : '#cdd6f4'};cursor:pointer;font-size:0.9rem;"
    >
      {label}
    </button>
  {/each}
  <span style="margin-left:auto">
    <button
      onclick={() => navigate('onboarding')}
      style="background:#89b4fa;border:none;color:#1e1e2e;padding:0.3rem 0.75rem;border-radius:4px;cursor:pointer;font-size:0.85rem;font-weight:bold;"
    >
      + New Benchmark
    </button>
  </span>
</nav>

<main
  style="padding:1rem;font-family:monospace;background:#1e1e2e;min-height:100vh;color:#cdd6f4;"
>
  {#if page === 'dashboard'}
    <Dashboard onRunSelect={(id: string) => navigate('run-detail', { runId: id })} />
  {:else if page === 'run-detail' && runDetailId}
    <RunDetail runId={runDetailId} />
  {:else if page === 'compare'}
    <Compare />
  {:else if page === 'models'}
    <Models />
  {:else if page === 'datasets'}
    <Datasets />
  {:else if page === 'optimize'}
    <Optimize />
  {:else if page === 'doctor'}
    <Doctor />
  {:else if page === 'onboarding'}
    <Onboarding onDone={() => navigate('dashboard')} />
  {/if}
</main>

<style>
  :global(body) {
    margin: 0;
    background: #1e1e2e;
  }
  :global(button) {
    cursor: pointer;
  }
</style>
