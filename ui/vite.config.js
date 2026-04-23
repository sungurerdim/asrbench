import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: '../asrbench/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/runs': 'http://localhost:8765',
      '/models': 'http://localhost:8765',
      '/datasets': 'http://localhost:8765',
      '/optimize': 'http://localhost:8765',
      '/system': 'http://localhost:8765',
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
      },
    },
  },
})
