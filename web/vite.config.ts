// vitest's defineConfig, not vite's — it is the one that knows the `test` key.
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// In development the API lives on a separate process (uvicorn), so /api and /ws
// are proxied. In production FastAPI serves web/dist directly and the same
// relative paths resolve without a proxy — one origin, no CORS either way.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true,
                rewrite: (p) => p.replace(/^\/api/, "") },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
