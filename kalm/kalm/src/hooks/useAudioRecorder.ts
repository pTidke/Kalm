// src/hooks/useAudioRecorder.ts
import { useEffect, useMemo, useRef, useState } from "react";
import {
  useAudioRecorder as useExpoAudioRecorder,
  useAudioRecorderState,
  RecordingPresets,
  setAudioModeAsync,
  AudioModule,
} from "expo-audio";

type RecState = {
  isRecording: boolean;
  isBusy: boolean;
  durationMs: number;
  uri: string | null;
  error: string | null;
  start: () => Promise<void>;
  stop: () => Promise<string | null>;
  toggle: () => Promise<string | null>;
};

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function useAudioRecorder(): RecState {
  // Recorder instance + live state (poll ~200ms for snappy UI updates)
  const recorder = useExpoAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const s = useAudioRecorderState(recorder, 200);

  // Local UI state
  const [error, setError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);

  // Refs for robustness
  const urlRef = useRef<string | null>(null);       // always the latest URL from state
  const lastTapRef = useRef<number>(0);             // debounce rapid taps

  useEffect(() => {
    urlRef.current = s.url ?? null;
  }, [s.url]);

  // Permissions + audio mode (once)
  useEffect(() => {
    (async () => {
      try {
        const perm = await AudioModule.requestRecordingPermissionsAsync();
        if (!perm.granted) {
          setError("Microphone permission denied");
          return;
        }
        await setAudioModeAsync({
          allowsRecording: true,
          playsInSilentMode: true,
        });
      } catch (e: any) {
        setError(e?.message ?? "Audio init failed");
      }
    })();
    // no cleanup required; expo-audio handles release on unmount
  }, []);

  async function start() {
    if (isBusy || s.isRecording) return;
    setError(null);
    setIsBusy(true);
    try {
      await recorder.prepareToRecordAsync();
      recorder.record();
    } catch (e: any) {
      setError(e?.message ?? "Failed to start recording");
    } finally {
      setIsBusy(false);
    }
  }

  async function stop(): Promise<string | null> {
    if (isBusy || !s.isRecording) return null;
    setIsBusy(true);
    try {
      await recorder.stop();

      // expo-audio may surface the URL slightly after stop() resolves.
      // Wait briefly (up to ~600ms) for the URL to appear.
      let uri: string | null = urlRef.current ?? (recorder as any)?.uri ?? null;
      for (let i = 0; i < 10 && !uri; i++) {
        await delay(60);
        uri = urlRef.current ?? (recorder as any)?.uri ?? null;
      }
      return uri ?? null;
    } catch (e: any) {
      setError(e?.message ?? "Failed to stop recording");
      return null;
    } finally {
      setIsBusy(false);
    }
  }

  async function toggle(): Promise<string | null> {
    // Debounce rapid taps to avoid double-activating during transitions
    const now = Date.now();
    if (now - lastTapRef.current < 300) return null;
    lastTapRef.current = now;

    if (isBusy) return null;
    return s.isRecording ? stop() : (await start(), null);
  }

  // Derived state for consumers
  const state = useMemo(
    () => ({
      isRecording: s.isRecording,
      isBusy,
      durationMs: s.durationMillis ?? 0,
      uri: urlRef.current,
      error,
    }),
    [s.isRecording, s.durationMillis, isBusy, error]
  );

  return { ...state, start, stop, toggle };
}
