import Constants from "expo-constants";
import { Platform } from "react-native";

const API_BASE =
  (Constants.expoConfig?.extra as any)?.API_BASE ??
  (Constants.manifest as any)?.extra?.API_BASE ??
  "http://localhost:3001";

export async function sendChat(messages: { role: string; content: string }[]) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) throw new Error("chat failed");
  return res.json();
}

export async function transcribeAudio(uri: string) {
  const form = new FormData();
  form.append("file", {
    uri,
    name: "audio.m4a",
    type: Platform.OS === "ios" ? "audio/m4a" : "audio/mp4",
  } as any);
  const res = await fetch(`${API_BASE}/transcribe`, {
    method: "POST",
    body: form,
    // NOTE: DO NOT set Content-Type; RN sets correct multipart boundary
  });
  if (!res.ok) throw new Error("transcribe failed");
  return res.json() as Promise<{ text: string }>;
}
