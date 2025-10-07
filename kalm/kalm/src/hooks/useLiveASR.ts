// src/hooks/useLiveASR.ts
import { useRef, useState, useEffect, useCallback } from "react";
import { asrOpenSession, asrAppend } from "../services/api";
import { useAudioRecorder } from "./useAudioRecorder";

type LiveASR = {
  isRecording: boolean;      // session-based (stable)
  isBusy: boolean;           // cutting/uploading a chunk
  partialText: string;       // rolling transcript
  error: string | null;
  start: () => Promise<void>;
  stop: () => Promise<string | null>; // returns final text
  toggle: () => Promise<string | null>;
};

export function useLiveASR(intervalMs = 1400): LiveASR {
  const rec = useAudioRecorder(); // must expose start() / stop()
  const [sid, setSid] = useState<string | null>(null);
  const [partialText, setPartialText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setBusy] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const seqRef = useRef(0);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const cutAndUpload = useCallback(async (final: boolean) => {
    if (!sid) return null;
    if (isBusy) return null;
    setBusy(true);
    try {
      const uri = await rec.stop();                  // 1) cut
      if (!uri) { setBusy(false); return null; }
      const res = await asrAppend(sid, seqRef.current++, uri, final); // 2) upload
      setPartialText(res.text);                      // 3) show rolling text
      if (!final) await rec.start();                 // 4) immediately resume
      setBusy(false);
      return res.text;
    } catch (e: any) {
      setBusy(false);
      setError(e?.message ?? "ASR upload failed");
      return null;
    }
  }, [sid, rec, isBusy]);

  const start = useCallback(async () => {
    if (sid) return; // already running
    setError(null);
    setPartialText("");
    seqRef.current = 0;
    const s = await asrOpenSession();
    setSid(s.sid);
    await rec.start();
    timerRef.current = setInterval(() => { void cutAndUpload(false); }, intervalMs);
  }, [sid, rec, intervalMs, cutAndUpload]);

  const stop = useCallback(async () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    let finalText: string | null = null;
    if (sid) finalText = await cutAndUpload(true);
    else await rec.stop(); // safety
    setSid(null);
    return finalText;
  }, [sid, cutAndUpload, rec]);

  const toggle = useCallback(async () => {
    if (sid) return await stop();
    await start();
    return null;
  }, [sid, start, stop]);

  return { isRecording: !!sid, isBusy, partialText, error, start, stop, toggle };
}
