import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  View, Text, FlatList, TextInput, Pressable, ActivityIndicator,
  KeyboardAvoidingView, Platform, Animated
} from "react-native";
import { RouteProp, useRoute } from "@react-navigation/native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
dayjs.extend(relativeTime);

import { useThreads } from "../store/threads";
import type { Message, RootStackParamList } from "../types";
import { streamChat, chatOnce, transcribeAudio } from "../services/api";
import Bubble from "../components/Bubble";
import { theme, fs } from "../ui/theme";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

type ChatRoute = RouteProp<RootStackParamList, "Chat">;

const MIN_LINE_H = 20;
const ROW_VERT_PAD = 8;

/** Assistant typing dots */
function TypingDots() {
  const a1 = useRef(new Animated.Value(0)).current;
  const a2 = useRef(new Animated.Value(0)).current;
  const a3 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const pulse = (av: Animated.Value, delay: number) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(av, { toValue: 1, duration: 300, useNativeDriver: true }),
          Animated.timing(av, { toValue: 0.3, duration: 300, useNativeDriver: true }),
          Animated.delay(150),
        ])
      ).start();
    pulse(a1, 0);
    pulse(a2, 150);
    pulse(a3, 300);
  }, [a1, a2, a3]);

  const Dot = ({ av }: { av: Animated.Value }) => (
    <Animated.View
      style={{
        width: 6, height: 6, borderRadius: 3, backgroundColor: theme.colors.subtext,
        marginHorizontal: 3, opacity: av,
        transform: [{ translateY: av.interpolate({ inputRange: [0, 1], outputRange: [0, -2] }) }],
      }}
    />
  );

  return (
    <View style={{ flexDirection: "row", alignItems: "center", paddingVertical: 2 }}>
      <Dot av={a1} /><Dot av={a2} /><Dot av={a3} />
    </View>
  );
}

/** Listening equalizer pill */
function ListeningIndicator({ visible }: { visible: boolean }) {
  const bars = [0, 1, 2, 3].map(() => useRef(new Animated.Value(0)).current);

  useEffect(() => {
    const loops: Animated.CompositeAnimation[] = [];
    if (visible) {
      bars.forEach((v, i) => {
        v.setValue(0.2);
        const loop = Animated.loop(
          Animated.sequence([
            Animated.timing(v, { toValue: 1, duration: 300 + i * 40, useNativeDriver: true }),
            Animated.timing(v, { toValue: 0.2, duration: 260 + i * 40, useNativeDriver: true }),
          ])
        );
        loop.start();
        loops.push(loop);
      });
    }
    return () => { loops.forEach((l) => l.stop?.()); bars.forEach(v => v.stopAnimation()); };
  }, [visible]);

  if (!visible) return null;

  const Bar = ({ av }: { av: Animated.Value }) => (
    <Animated.View
      style={{
        width: 4, marginHorizontal: 3, borderRadius: 2, backgroundColor: theme.colors.accent, height: 12,
        transform: [{ scaleY: av.interpolate({ inputRange: [0, 1], outputRange: [0.6, 1.4] }) }],
      }}
    />
  );

  return (
    <View
      style={{
        alignSelf: "center",
        flexDirection: "row", alignItems: "center", gap: 8,
        paddingVertical: 10, paddingHorizontal: 10,
        borderRadius: theme.radius.pill, borderWidth: 1, borderColor: theme.colors.accent,
        backgroundColor: theme.colors.surface,
        marginBottom: theme.spacing.sm,
      }}
    >
      <View style={{ flexDirection: "row", alignItems: "flex-end", marginRight: 4 }}>
        {bars.map((b, i) => <Bar key={i} av={b} />)}
      </View>
      <Text style={{ color: theme.colors.accent, fontWeight: "600", fontSize: fs(12) }}>Listening...</Text>
    </View>
  );
}

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const route = useRoute<ChatRoute>();
  const id = String(route.params?.id || "");
  const seed = route.params?.seed;

  const { threads, appendMessage } = useThreads();
  const thread = threads.find((t) => String(t.id) === id);

  const listRef = useRef<FlatList<Message>>(null);
  const [value, setValue] = useState("");
  const [typing, setTyping] = useState(false); // dots until first token
  const [focused, setFocused] = useState(false);
  const [inputHeight, setInputHeight] = useState(MIN_LINE_H);

  const rec = useAudioRecorder(); // has isRecording, isBusy, toggle(), error

  const [msgs, setMsgs] = useState<Message[]>([]);
  const assistantBuf = useRef<string>("");
  const streamCloser = useRef<{ close: () => void } | null>(null);
  const seededRef = useRef(false);

  useEffect(() => {
    if (thread && msgs.length === 0) setMsgs(thread.messages ?? []);
  }, [thread]); // eslint-disable-line

  useEffect(() => {
    if (!thread) return;
    const storeMsgs = thread.messages ?? [];
    if (storeMsgs.length > msgs.length) setMsgs(storeMsgs);
  }, [thread?.messages, thread?.updatedAt]); // eslint-disable-line

  // Server convo (includes any system), UI hides system
  const dataAll = useMemo(() => [...msgs].sort((a, b) => a.createdAt - b.createdAt), [msgs]);
  const dataUI  = useMemo(() => dataAll.filter((m) => m.role !== "system"), [dataAll]);

  const scrollToEnd = () => setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 16);

  if (!thread) {
    return (
      <View style={{ flex: 1, backgroundColor: theme.colors.bg, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator />
        <Text style={{ color: theme.colors.subtext, marginTop: 12, fontSize: theme.type.body }}>Loading chat…</Text>
      </View>
    );
  }

  // Seed first message if provided and no visible messages yet
  useEffect(() => {
    if (seed && !seededRef.current && dataUI.length === 0) {
      seededRef.current = true;
      (async () => { await send(seed); })();
    }
  }, [seed, dataUI.length]); // eslint-disable-line

  /** Start streaming: show dots until first token, then live-update one assistant bubble */
  const startStreaming = async (
    convo: { role: "assistant" | "user" | "system"; content: string }[],
    assistantId: string
  ) => {
    setTyping(true);
    assistantBuf.current = "";
    streamCloser.current?.close();

    let gotFirst = false;

    const closer = streamChat(
      convo,
      {
        onDelta: (chunk) => {
          assistantBuf.current += chunk;

          if (!gotFirst) {
            gotFirst = true;
            setTyping(false); // swap dots → bubble immediately
            const draft: Message = {
              id: assistantId,
              role: "assistant",
              text: assistantBuf.current,
              createdAt: Date.now(),
            };
            setMsgs((prev) => [...prev, draft]);
            scrollToEnd();
          } else {
            setMsgs((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: assistantBuf.current } : m))
            );
          }
        },
        onDone: async (info) => {
          if (!gotFirst) {
            // No tokens arrived — fallback once to HTTP
            try {
              const { reply } = await chatOnce(convo as any);
              const finalMsg: Message = {
                id: assistantId, role: "assistant",
                text: (reply || "Sorry—no response right now.").trim(),
                createdAt: Date.now(), intent: info?.intent,
              };
              setMsgs((prev) => [...prev, finalMsg]);
              appendMessage(thread.id, finalMsg).catch(() => {});
              setTyping(false);
              scrollToEnd();
              return;
            } catch {
              const finalMsg: Message = {
                id: assistantId, role: "assistant",
                text: "Sorry—no response right now.",
                createdAt: Date.now(),
              };
              setMsgs((prev) => [...prev, finalMsg]);
              appendMessage(thread.id, finalMsg).catch(() => {});
              setTyping(false);
              scrollToEnd();
              return;
            }
          }

          const trimmed = assistantBuf.current.trim();
          setMsgs((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, text: trimmed, intent: info?.intent } : m))
          );
          appendMessage(thread.id, {
            id: assistantId, role: "assistant", text: trimmed, createdAt: Date.now(), intent: info?.intent,
          }).catch(() => {});
          scrollToEnd();
        },
        onError: async (err) => {
          if (!gotFirst) {
            try {
              const { reply } = await chatOnce(convo as any);
              const finalMsg: Message = {
                id: assistantId, role: "assistant",
                text: (reply || "Sorry—no response right now.").trim(),
                createdAt: Date.now(),
              };
              setMsgs((prev) => [...prev, finalMsg]);
              appendMessage(thread.id, finalMsg).catch(() => {});
              setTyping(false);
              scrollToEnd();
              return;
            } catch {}
          }
          const finalMsg: Message = {
            id: assistantId, role: "assistant",
            text: assistantBuf.current || "Sorry—streaming failed. Please try again.",
            createdAt: Date.now(),
          };
          setMsgs((prev) => prev.map((m) => (m.id === assistantId ? finalMsg : m)));
          appendMessage(thread.id, finalMsg).catch(() => {});
          setTyping(false);
          scrollToEnd();
          console.warn("ws stream error:", err);
        },
      },
      { seed: undefined }
    );

    streamCloser.current = closer;
  };

  /** Send text (from keyboard or seeded), streaming reply */
  const send = async (overrideText?: string) => {
    const text = (overrideText ?? value).trim();
    if (!text || typing) return;

    const userMsg: Message = { id: Math.random().toString(), role: "user", text, createdAt: Date.now() };
    setMsgs((prev) => [...prev, userMsg]);
    appendMessage(thread.id, userMsg).catch(() => {});
    if (!overrideText) setValue("");
    scrollToEnd();

    const assistantId = "a-" + Math.random().toString().slice(2);
    const convo = [...dataAll, userMsg].map((m) => ({
      role: m.role as "user" | "assistant" | "system",
      content: m.text,
    })) as { role: "assistant" | "user" | "system"; content: string }[];

    await startStreaming(convo, assistantId);
  };

  /** Dots bubble (footer) shown before first token */
  const TypingBubble = () => (
    <View
      style={{
        alignSelf: "flex-start",
        backgroundColor: theme.colors.bubbleAI,
        paddingVertical: 10,
        paddingHorizontal: 14,
        borderRadius: theme.radius.lg,
        marginVertical: 6,
        borderWidth: 1,
        borderColor: theme.colors.line,
      }}
    >
      <TypingDots />
    </View>
  );

  /** Mic: toggle record; when stopped, transcribe → put result into the input (do NOT auto-send) */
  const MicButton = () => (
    <Pressable
      onPress={async () => {
        if (rec.isBusy) return;
        const uri = await rec.toggle();
        if (uri) {
          try {
            const { text } = await transcribeAudio(uri);
            if (text?.trim()) {
              // Put transcript into composer (append with a space if needed)
              setValue((prev) => (prev ? `${prev} ${text.trim()}` : text.trim()));
            }
          } catch (e) {
            console.warn("transcribe error", e);
          }
        }
      }}
      disabled={rec.isBusy}
      style={{
        opacity: rec.isBusy ? 0.6 : 1,
        backgroundColor: rec.isRecording ? "#ef4444" : theme.colors.accent,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: theme.radius.pill,
      }}
      hitSlop={12}
    >
      <View style={{ width: 16, height: 16, borderRadius: 8, backgroundColor: "white" }} />
    </Pressable>
  );

  const composerHeight = inputHeight + ROW_VERT_PAD * 2;
  const listeningExtra = rec.isRecording ? (theme.spacing.sm + 28) : 0;
  const listBottomPadding = composerHeight + theme.spacing.xl + insets.bottom + listeningExtra;

  const canSend = !!value.trim() && !typing;
  const isIdleDisabled = !value.trim() && !typing;

  return (
    <View style={{ flex: 1, backgroundColor: theme.colors.bg }}>
      <View style={{ height: insets.top, backgroundColor: theme.colors.surface }} />

      {/* Header */}
      <View
        style={{
          paddingHorizontal: theme.spacing.xl,
          paddingTop: theme.spacing.lg,
          paddingBottom: theme.spacing.md,
          borderBottomWidth: 1,
          borderBottomColor: theme.colors.line,
          backgroundColor: theme.colors.surface,
        }}
      >
        <Text style={{ color: theme.colors.text, fontSize: theme.type.title, fontWeight: "800" }}>
          {thread.title}
        </Text>
        <Text style={{ color: theme.colors.subtext, fontSize: theme.type.small, marginTop: 6 }}>
          Not medical advice. In emergencies call local services.
        </Text>
      </View>

      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.select({ ios: "padding", android: undefined })}>
        <FlatList
          ref={listRef}
          data={dataUI}
          keyExtractor={(i) => i.id}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={{
            paddingHorizontal: theme.spacing.lg,
            paddingTop: theme.spacing.lg,
            paddingBottom: listBottomPadding,
          }}
          renderItem={({ item }) => <Bubble msg={item} />}
          ListFooterComponent={typing ? <TypingBubble /> : <View style={{ height: 0 }} />}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          extraData={[dataUI, listBottomPadding, typing, rec.isRecording]}
        />

        {/* Composer */}
        <View
          style={{
            paddingHorizontal: theme.spacing.lg,
            paddingBottom: theme.spacing.lg + insets.bottom,
            backgroundColor: theme.colors.bg,
          }}
        >
          <ListeningIndicator visible={rec.isRecording} />

          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              gap: theme.spacing.md,
              backgroundColor: theme.colors.surface,
              borderRadius: theme.radius.xl,
              borderWidth: 1,
              borderColor: theme.colors.line,
              paddingHorizontal: theme.spacing.md,
              paddingVertical: ROW_VERT_PAD,
              ...theme.shadow(6),
            }}
          >
            <MicButton />

            <View style={{ flex: 1, justifyContent: "center" }}>
              {!value && !focused && (
                <View
                  pointerEvents="none"
                  style={{ position: "absolute", left: 0, right: 0, top: 0, bottom: 0, justifyContent: "center" }}
                >
                  <Text numberOfLines={1} style={{ color: theme.colors.subtext, fontSize: theme.type.body }}>
                    Type your thoughts…
                  </Text>
                </View>
              )}

              <TextInput
                value={value}
                onChangeText={setValue}
                multiline
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                onSubmitEditing={() => send()}
                onContentSizeChange={(e) => {
                  const h = Math.ceil(e.nativeEvent.contentSize.height || MIN_LINE_H);
                  setInputHeight(Math.max(MIN_LINE_H, Math.min(120, h)));
                }}
                style={{
                  height: inputHeight,
                  color: theme.colors.text,
                  fontSize: theme.type.body,
                  lineHeight: MIN_LINE_H,
                  paddingTop: 0,
                  paddingBottom: 0,
                  margin: 0,
                  textAlignVertical: "center",
                  // @ts-ignore android-only
                  includeFontPadding: false,
                }}
              />
            </View>

            <Pressable
              onPress={() => send()}
              disabled={!canSend}
              style={[
                {
                  paddingVertical: 8,
                  paddingHorizontal: 14,
                  borderRadius: theme.radius.pill,
                },
                isIdleDisabled
                  ? { backgroundColor: "transparent", borderWidth: 1, borderColor: theme.colors.accent }
                  : { backgroundColor: theme.colors.accent, ...theme.shadow(4) },
              ]}
            >
              {typing ? (
                <ActivityIndicator color="#0b1220" />
              ) : (
                <Text
                  style={{
                    color: isIdleDisabled ? theme.colors.accent : "#0b1220",
                    fontWeight: "800",
                    fontSize: fs(14),
                  }}
                >
                  Send
                </Text>
              )}
            </Pressable>
          </View>

          {rec.error ? (
            <Text
              style={{
                color: "#ef4444",
                fontSize: theme.type.small,
                marginTop: theme.spacing.sm,
                textAlign: "center",
              }}
            >
              Mic error: {rec.error}
            </Text>
          ) : null}

          <Text
            style={{
              color: theme.colors.subtext,
              fontSize: theme.type.micro,
              marginTop: theme.spacing.sm,
              alignSelf: "center",
            }}
          >
            Last updated {dayjs(thread.updatedAt).fromNow()}
          </Text>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}
