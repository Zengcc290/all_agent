import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

// 构建产物直接落到 web/static，FastAPI 挂载的就是这个目录：
// `python -m web.app` 不需要额外起 Vite 就能打开前端。
// 开发态用 `npm run dev`，由下面的 proxy 把 /api 转发到 FastAPI。
export default defineConfig({
  plugins: [react()],
  root: fileURLToPath(new URL('.', import.meta.url)),
  build: { outDir: '../static', emptyOutDir: true, sourcemap: false, chunkSizeWarningLimit: 900 },
  server: {
    port: 5173,
    // 不设 host 时 vite5 在部分平台只绑 IPv6 的 ::1，导致 http://127.0.0.1:5173 连不上
    host: '0.0.0.0',
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
})