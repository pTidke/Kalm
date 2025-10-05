import { useState } from "react";
import { Audio } from "expo-av";

// Read constants in a TS-safe way across SDKs
const A: any = Audio;

const FALLBACK_OPTIONS: Audio.RecordingOptions = {
  android: {
    extension: ".m4a",
    // fall back to known numeric values if constants don't exist
    outputFormat: A.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_MPEG_4 ?? 2,
    audioEncoder: A.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_AAC ?? 3,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 96000,
  },
  ios: {
    extension: ".m4a",
    audioQuality: A.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH ?? 0,
    outputFormat: A.RECORDING_OPTION_IOS_OUTPUT_FORMAT_MPEG4AAC ?? 2,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 96000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: {
    mimeType: "audio/webm",
    bitsPerSecond: 96000,
  },
};

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    try {
      setError(null);

      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) throw new Error("Microphone permission not granted");

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        // keep Android defaults simple for widest compatibility
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const rec = new Audio.Recording();

      // Prefer new preset API if present; else use our fallback options
      const preset =
        A?.RecordingOptionsPresets?.HIGH_QUALITY ?? FALLBACK_OPTIONS;

      await rec.prepareToRecordAsync(preset);
      await rec.startAsync();

      setRecording(rec);
      setIsRecording(true);
    } catch (e: any) {
      setError(e?.message ?? "Failed to start recording");
      setIsRecording(false);
      setRecording(null);
    }
  }

  async function stop(): Promise<string | null> {
    try {
      if (!recording) return null;
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setIsRecording(false);
      setRecording(null);
      return uri ?? null;
    } catch (e: any) {
      setError(e?.message ?? "Failed to stop recording");
      setIsRecording(false);
      setRecording(null);
      return null;
    }
  }

  return { isRecording, start, stop, error };
}
