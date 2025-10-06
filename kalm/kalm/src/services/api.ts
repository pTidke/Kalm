// src/services/api.ts
import Constants from "expo-constants";

const API_BASE = (Constants?.expoConfig?.extra as any)?.API_BASE || "http://localhost:3001";

export async function sendChat(messages: { role: string; content: string }[]) {
  const r = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function transcribeAudio(uri: string) {
  const fileName = uri.split("/").pop() || "audio.m4a";
  const type = "audio/m4a";
  const body = new FormData();
  // @ts-ignore RN FormData file
  body.append("file", { uri, name: fileName, type });
  const r = await fetch(`${API_BASE}/transcribe`, { method: "POST", body });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function wsUrl(path: string) {
  if (API_BASE.startsWith("https://")) return API_BASE.replace(/^https:\/\//, "wss://") + path;
  return API_BASE.replace(/^http:\/\//, "ws://") + path;
}

type StreamHandlers = {
  onDelta?: (text: string) => void;
  onDone?: (info: { intent?: string; topic?: string }) => void;
  onError?: (err: string) => void;
};

export function streamChat(
  messages: { role: string; content: string }[],
  handlers: StreamHandlers = {},
  opts?: { seed?: number }
) {
  const url = wsUrl("/ws/chat");
  const ws = new WebSocket(url);

  ws.onopen = () => {
    ws.send(JSON.stringify({ messages, seed: opts?.seed }));
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "delta") handlers.onDelta?.(msg.text);
      else if (msg.type === "done") handlers.onDone?.({ intent: msg.intent, topic: msg.topic });
      else if (msg.type === "error") handlers.onError?.(msg.error || "stream error");
    } catch {
      // ignore non-JSON
    }
  };

  ws.onerror = (e: any) => {
    handlers.onError?.(e?.message || "ws error");
  };

  ws.onclose = () => {
    // noop
  };

  return {
    close: () => ws.close(),
    socket: ws,
  };
}
