import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API to `python -m vidbush serve` on port 8080.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8080" },
  },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
});
