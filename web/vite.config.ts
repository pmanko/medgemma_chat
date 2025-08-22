import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import sveltePreprocess from 'svelte-preprocess';

export default defineConfig({
  plugins: [svelte({ preprocess: sveltePreprocess({ typescript: true }) })],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://localhost:8080',
      '/generate': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/manifest': 'http://localhost:8080'
    }
  },
  build: {
    outDir: 'dist'
  }
});


