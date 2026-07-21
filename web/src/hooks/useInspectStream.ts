import { useCallback, useRef, useState } from "react";
import {
  type InspectMessage,
  type InspectStart,
  type InspectToken,
  WS_BASE,
  parseAttention,
} from "../api/ws";

export interface InspectRequest {
  prompt: string;
  max_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  seed?: number;
  layers?: number[];
  heads?: number[];
  top_n?: number;
}

export interface InspectState {
  running: boolean;
  start: InspectStart | null;
  tokens: InspectToken[];
  error: string | null;
  bytesReceived: number;
}

const EMPTY: InspectState = {
  running: false,
  start: null,
  tokens: [],
  error: null,
  bytesReceived: 0,
};

/**
 * Drive WS /ws/inspect.
 *
 * The server sends a JSON token frame and then, if it declared an `attn` block,
 * the raw float32 payload. `pending` holds the frame awaiting its binary half —
 * the frames are strictly ordered, so a single slot is enough.
 */
export function useInspectStream() {
  const [state, setState] = useState<InspectState>(EMPTY);
  const socketRef = useRef<WebSocket | null>(null);
  const pendingRef = useRef<InspectToken | null>(null);

  const stop = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    pendingRef.current = null;
    setState((prev) => ({ ...prev, running: false }));
  }, []);

  const run = useCallback(
    (request: InspectRequest) => {
      socketRef.current?.close();
      pendingRef.current = null;
      setState({ ...EMPTY, running: true });

      const socket = new WebSocket(`${WS_BASE}/ws/inspect`);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => socket.send(JSON.stringify(request));

      socket.onmessage = (event: MessageEvent) => {
        if (event.data instanceof ArrayBuffer) {
          const frame = pendingRef.current;
          pendingRef.current = null;
          if (!frame?.attn) return;
          try {
            frame.attention = parseAttention(frame.attn.shape, event.data);
          } catch (err) {
            setState((prev) => ({ ...prev, error: (err as Error).message }));
            return;
          }
          setState((prev) => ({
            ...prev,
            tokens: [...prev.tokens, frame],
            bytesReceived: prev.bytesReceived + event.data.byteLength,
          }));
          return;
        }

        const message = JSON.parse(event.data as string) as InspectMessage;
        switch (message.type) {
          case "start":
            setState((prev) => ({ ...prev, start: message }));
            break;
          case "token":
            // Hold frames that declare attention until their binary half lands,
            // so a token never renders with the previous token's weights.
            if (message.attn) pendingRef.current = message;
            else setState((prev) => ({ ...prev, tokens: [...prev.tokens, message] }));
            break;
          case "done":
            setState((prev) => ({ ...prev, running: false }));
            break;
          case "error":
            setState((prev) => ({ ...prev, running: false, error: message.message }));
            break;
        }
      };

      socket.onerror = () =>
        setState((prev) => ({ ...prev, running: false, error: "websocket error" }));
      socket.onclose = () => setState((prev) => ({ ...prev, running: false }));
    },
    [],
  );

  return { ...state, run, stop };
}
