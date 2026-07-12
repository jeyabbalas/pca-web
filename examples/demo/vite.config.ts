import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs so the static build works at any GitHub Pages path.
  base: './',
  // The worker entry imports ESM ('pca-web/worker'); the default iife
  // worker format cannot represent that.
  worker: { format: 'es' },
  // pca-web is a workspace symlink; serve its real (pure-ESM) files in dev
  // instead of prebundling, so worker/client subpaths resolve untouched.
  optimizeDeps: { exclude: ['pca-web'] },
  build: { target: 'es2022' },
});
