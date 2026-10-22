import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' emits relative asset paths, so the same build works on Vercel
// (served from /) and on GitHub Pages (served from /<repo-name>/).
export default defineConfig({
  plugins: [react()],
  base: './',
});
