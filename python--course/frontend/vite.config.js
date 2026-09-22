import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 不设 host 时 vite5 只绑 IPv6 的 ::1，导致 http://127.0.0.1:5173 连不上
    host: '0.0.0.0',
    strictPort: true,
    // 开发态把 /api 代理到 fastapi 后端
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 900 },
})
