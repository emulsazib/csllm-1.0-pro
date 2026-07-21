/** Thin fetch wrapper around the gateway.
 *
 *  Every call goes through `request`, so the FastAPI error shape (`{detail}`) is
 *  turned into a real Error in exactly one place — otherwise a 422 would surface
 *  in the UI as "[object Object]".
 */

import type {
  EmbeddingsResponse,
  Health,
  InspectResponse,
  SamplingSettings,
  TokenizeResponse,
} from "./types";

/** Where the API lives, which differs by build mode:
 *
 *  · dev  — the app is on Vite (5173) and the gateway on another port, so calls
 *           go to `/api/...` and vite.config.ts proxies them across, stripping
 *           the prefix.
 *  · prod — FastAPI serves this bundle itself, so the API is same-origin at the
 *           root and there is no proxy to strip anything.
 *
 *  Hard-coding "/api" made the built app request `/api/tokenize`, which hits the
 *  StaticFiles mount instead of the route — "Method Not Allowed" on every call.
 *  The server surface stays at the root either way; only the client prefix moves.
 */
export const API_BASE = import.meta.env.DEV ? "/api" : "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function detailOf(payload: unknown, fallback: string): string {
  if (typeof payload !== "object" || payload === null) return fallback;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  // Pydantic validation errors arrive as a list of {loc, msg}.
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        const loc = Array.isArray(d?.loc) ? d.loc.slice(1).join(".") : "";
        return loc ? `${loc}: ${d?.msg}` : String(d?.msg ?? "");
      })
      .filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(
      detailOf(payload, `${response.status} ${response.statusText}`),
      response.status,
    );
  }
  return (await response.json()) as T;
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body), signal });
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal }),

  tokenize: (text: string, signal?: AbortSignal) =>
    post<TokenizeResponse>("/tokenize", { text }, signal),

  embeddings: (text: string, project = true, signal?: AbortSignal) =>
    post<EmbeddingsResponse>("/embeddings", { text, project }, signal),

  inspect: (prompt: string, settings: SamplingSettings, topN = 20, signal?: AbortSignal) =>
    post<InspectResponse>(
      "/inspect/next_token",
      { prompt, ...settings, top_n: topN },
      signal,
    ),
};
