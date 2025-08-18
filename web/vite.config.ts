import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import sveltePreprocess from 'svelte-preprocess';

export default defineConfig({
  plugins: [svelte({ preprocess: sveltePreprocess({ typescript: true }) })],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://localhost:3001',
      '/generate': 'http://localhost:3001',
      '/health': 'http://localhost:3001',
      '/manifest': 'http://localhost:3001'
    }
  },
  build: {
    outDir: 'dist'
  }
});


