import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      // 后端 FastAPI (server.py, 端口 8001)
      '/health': 'http://127.0.0.1:8001',
      '/resolve': 'http://127.0.0.1:8001',
      // 播放轨: 直链流式代理 (透传 Range, 支持拖动)
      '/stream': 'http://127.0.0.1:8001',
      '/feedback': 'http://127.0.0.1:8001',
      // 推理轨: YouTube 双轨 WebSocket + 摄像头推帧 WebSocket
      '/ws': {
        target: 'ws://127.0.0.1:8001',
        ws: true,
      },
    },
  },
})
