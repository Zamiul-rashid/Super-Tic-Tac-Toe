import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The bundle is served by the FastAPI process itself, which mounts
// sttt/web/static/assets at /assets and returns index.html at /.
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../sttt/web/static', emptyOutDir: true, assetsDir: 'assets' },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
