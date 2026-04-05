import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/** Backend origin for dev proxy. Override if port 8000 is blocked (e.g. WinError 10013): use 8080. */
function apiProxyTarget(mode) {
  const env = loadEnv(mode, process.cwd(), "");
  return env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";
}

export default defineConfig(({ mode }) => {
  const target = apiProxyTarget(mode);

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/predict-crop": target,
        "/predict-yield": target,
        "/recommend": target,
        "/analytics": target,
        "/health": target,
      },
    },
  };
});
