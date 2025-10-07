// src/services/api.ts
import Constants from "expo-constants";

export type ChatMsg = { role: "system" | "user" | "assistant"; content: string };

const API_BASE: string =
  (Constants?.expoConfig?.extra as any)?.API_BASE ||
  (Constants?.manifest?.extra as any)?.API_BASE ||
  "http://127.0.0.1:3001";

// ---- HTTP helper ----
async function http<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

// ---- Whisper transcription ----
export async function transcribeAudio(fileUri: string): Promise<{ text: string }> {
  const form = new FormData();
  // Best-effort content type; server just needs a file
  form.append(
    "file",
    {
      uri: fileUri,
      name: "input.m4a",
      type: "audio/m4a",
    } as any
  );

  const res = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: form as any });
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(t || "transcribe failed");
  }
  return (await res.json()) as { text: string };
}

// ---- Non-stream fallback (used if stream yields zero tokens) ----
export async function chatOnce(messages: ChatMsg[]): Promise<{ reply: string; intent?: string }> {
  return http<{ reply: string; intent?: string }>("/chat", {
    method: "POST",
    body: JSON.stringify({ messages }),
  });
}

// ---- Streaming over WebSocket ----
export function streamChat(
  messages: ChatMsg[],
  handlers: {
    onDelta: (chunk: string) => void;
    onDone: (info: { intent?: string }) => void;
    onError: (err: any) => void;
  },
  opts?: { seed?: number }
): { close: () => void } {
  // Convert http -> ws
  const wsUrl = `${API_BASE.replace(/^http/, "ws")}/ws/chat`;

  // Simple whitespace fixer so "Hi" + "there" doesn't render "Hithere"
  const needSpace = (prev: string, next: string) => {
    if (!prev || !next) return false;
    const a = prev[prev.length - 1];
    const b = next[0];
    return /[A-Za-z0-9)]/.test(a) && /[A-Za-z]/.test(b); // wordy boundary
  };
  const normalizeDelta = (prev: string, raw: any): string => {
    const d = typeof raw === "string" ? raw : raw?.toString?.() ?? "";
    if (!d) return "";
    // avoid collapsing existing whitespace, just prefix a single space when needed
    if (needSpace(prev, d)) return " " + d;
    return d;
  };

  let closed = false;
  let ws: WebSocket | null = null;
  let bufferSoFar = "";

  // Watchdog: if we never receive a delta in N seconds, give up.
  const watchdogMs = 15000;
  let watchdog: any = setTimeout(() => {
    try {
      ws?.close();
    } catch {}
    if (!closed) handlers.onError(new Error("stream timeout"));
  }, watchdogMs);

  function clearWatchdog() {
    if (watchdog) {
      clearTimeout(watchdog);
      watchdog = null;
    }
  }

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      // Send initial message payload
      ws?.send(
        JSON.stringify({
          type: "start",
          messages,
          seed: opts?.seed,
        })
      );
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(typeof ev.data === "string" ? ev.data : "");
        if (msg?.type === "delta") {
          const chunk = normalizeDelta(bufferSoFar, msg.text || "");
          if (chunk) {
            bufferSoFar += chunk;
            handlers.onDelta(chunk);
          }
        } else if (msg?.type === "done") {
          clearWatchdog();
          handlers.onDone({ intent: msg.intent });
          try {
            ws?.close();
          } catch {}
        } else if (msg?.type === "error") {
          clearWatchdog();
          handlers.onError(new Error(msg.message || "stream error"));
          try {
            ws?.close();
          } catch {}
        }
      } catch (e) {
        // If server sent raw text chunks (not JSON), treat it as a delta
        const asText = String(ev.data ?? "");
        if (asText) {
          const chunk = normalizeDelta(bufferSoFar, asText);
          bufferSoFar += chunk;
          handlers.onDelta(chunk);
        }
      }
    };

    ws.onerror = (e) => {
      clearWatchdog();
      handlers.onError(e);
    };

    ws.onclose = () => {
      clearWatchdog();
      closed = true;
    };
  } catch (e) {
    clearWatchdog();
    handlers.onError(e);
  }

  return {
    close: () => {
      closed = true;
      try {
        ws?.close();
      } catch {}
    },
  };
}
