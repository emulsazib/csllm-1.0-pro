import { afterEach, describe, expect, it, vi } from "vitest";
import { API_BASE, ApiError, api } from "./client";

function mockFetch(status: number, body: unknown, ok = status < 400) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status,
    statusText: "Error",
    json: async () => body,
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("posts JSON to the gateway path for this build mode", async () => {
    const fetchMock = mockFetch(200, { tokens: [], count: 0 });
    await api.tokenize("hello");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}/tokenize`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ text: "hello" });
  });

  it("only prefixes /api in dev, where the Vite proxy strips it", () => {
    // In a production build FastAPI serves this bundle itself and the API is
    // same-origin at the root; a hard-coded /api hit the StaticFiles mount and
    // returned "Method Not Allowed" on every call.
    expect(API_BASE).toBe(import.meta.env.DEV ? "/api" : "");
    expect(API_BASE.endsWith("/")).toBe(false);
  });

  it("flattens sampling settings into the inspect payload", async () => {
    const fetchMock = mockFetch(200, {});
    await api.inspect("prompt", { temperature: 0.5, top_k: 10, top_p: 0.9 }, 25);

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      prompt: "prompt",
      temperature: 0.5,
      top_k: 10,
      top_p: 0.9,
      top_n: 25,
    });
  });

  it("surfaces a string detail from the gateway", async () => {
    mockFetch(422, { detail: "head_dim (5) must be even for RoPE pairing" });
    await expect(api.tokenize("x")).rejects.toThrow(/even for RoPE/);
  });

  it("flattens Pydantic validation errors into a readable message", async () => {
    // Without this the UI would render "[object Object]".
    mockFetch(422, {
      detail: [
        { loc: ["body", "temperature"], msg: "Input should be less than or equal to 2" },
        { loc: ["body", "top_p"], msg: "Input should be greater than 0" },
      ],
    });
    await expect(api.tokenize("x")).rejects.toThrow(
      "temperature: Input should be less than or equal to 2; top_p: Input should be greater than 0",
    );
  });

  it("falls back to the status line when the body is not JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    });
    vi.stubGlobal("fetch", fetchMock);
    await expect(api.health()).rejects.toThrow("500 Internal Server Error");
  });

  it("exposes the status code on ApiError", async () => {
    mockFetch(503, { detail: "model is not loaded" });
    await expect(api.health()).rejects.toMatchObject({
      name: "ApiError",
      status: 503,
    });
    expect(new ApiError("x", 404).status).toBe(404);
  });
});
