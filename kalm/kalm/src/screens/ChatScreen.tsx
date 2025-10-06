// src/screens/ChatScreen.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  View,
  Text,
  FlatList,
  TextInput,
  Pressable,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Animated,
} from "react-native";
import { RouteProp, useRoute } from "@react-navigation/native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
dayjs.extend(relativeTime);

import { useThreads } from "../store/threads";
import type { Message, RootStackParamList } from "../types";
import { streamChat, transcribeAudio } from "../services/api";
import Bubble from "../components/Bubble";
import { theme, fs } from "../ui/theme";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

type ChatRoute = RouteProp<RootStackParamList, "Chat">;

const MIN_LINE_H = 20; // pairs with smaller body font

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
        width: 6,
        height: 6,
        borderRadius: 3,
        backgroundColor: theme.colors.subtext,
        marginHorizontal: 3,
        opacity: av,
        transform: [
          {
            translateY: av.interpolate({
              inputRange: [0, 1],
              outputRange: [0, -2],
            }),
          },
        ],
      }}
    />
  );

  return (
    <View style={{ flexDirection: "row", alignItems: "center", paddingVertical: 2 }}>
      <Dot av={a1} />
      <Dot av={a2} />
      <Dot av={a3} />
    </View>
  );
}

export default function ChatScreen() {
  const insets = useSafeAreaInsets();
  const route = useRoute<ChatRoute>();
  const id = String(route.params?.id || "");

  const { threads, appendMessage } = useThreads();
  const thread = threads.find((t) => String(t.id) === id);

  const listRef = useRef<FlatList<Message>>(null);
  const [value, setValue] = useState("");
  const [sending, setSending] = useState(false);
  const [typing, setTyping] = useState(false);
  const [focused, setFocused] = useState(false);

  const [inputHeight, setInputHeight] = useState(MIN_LINE_H);
  const ROW_VERT_PAD = 8;

  const rec = useAudioRecorder();

  const [msgs, setMsgs] = useState<Message[]>([]);
  const assistantBufferRef = useRef<string>("");
  const streamCloser = useRef<{ close: () => void } | null>(null);

  useEffect(() => {
    if (thread && msgs.length === 0) setMsgs(thread.messages ?? []);
  }, [thread]); // eslint-disable-line

  useEffect(() => {
    if (!thread) return;
    const storeMsgs = thread.messages ?? [];
    if (storeMsgs.length > msgs.length) setMsgs(storeMsgs);
  }, [thread?.messages, thread?.updatedAt]); // eslint-disable-line

  const data = useMemo(
    () => [...msgs].sort((a, b) => a.createdAt - b.createdAt),
    [msgs]
  );

  const scrollToEnd = () =>
    setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 16);

  if (!thread) {
    return (
      <View style={{ flex: 1, backgroundColor: theme.colors.bg, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator />
        <Text style={{ color: theme.colors.subtext, marginTop: 12, fontSize: theme.type.body }}>
          Loading chat…
        </Text>
      </View>
    );
  }

  // --- STREAMING ---
  const startStreaming = (
    convo: { role: string; content: string }[],
    assistantId: string
  ) => {
    setTyping(true);
    assistantBufferRef.current = "";

    const closer = streamChat(
      convo,
      {
        onDelta: (chunk) => {
          assistantBufferRef.current += chunk;
          setMsgs((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, text: assistantBufferRef.current } : m
            )
          );
          scrollToEnd();
        },
        onDone: (info) => {
          setTyping(false);
          const finalText = assistantBufferRef.current || "…";
          const finalized: Message = {
            id: assistantId,
            role: "assistant",
            text: finalText,
            createdAt: Date.now(),
            intent: info.intent,
          };
          appendMessage(thread.id, finalized).catch(() => {});
        },
        onError: (err) => {
          setTyping(false);
          const msg: Message = {
            id: assistantId,
            role: "assistant",
            text: "Sorry—streaming failed. Please try again.",
            createdAt: Date.now(),
          };
          setMsgs((prev) => prev.map((m) => (m.id === assistantId ? msg : m)));
          appendMessage(thread.id, msg).catch(() => {});
          console.warn("ws stream error:", err);
        },
      },
      { seed: undefined }
    );
    streamCloser.current = closer;
  };

  const send = async (overrideText?: string) => {
    const text = (overrideText ?? value).trim();
    if (!text || sending || typing) return;

    const userMsg: Message = {
      id: Math.random().toString(),
      role: "user",
      text,
      createdAt: Date.now(),
    };

    setMsgs((prev) => [...prev, userMsg]);
    appendMessage(thread.id, userMsg).catch(() => {});
    if (!overrideText) setValue("");
    scrollToEnd();

    const assistantId = "a-" + Math.random().toString().slice(2);
    const draft: Message = {
      id: assistantId,
      role: "assistant",
      text: "",
      createdAt: Date.now(),
    };
    setMsgs((prev) => [...prev, draft]);
    scrollToEnd();

    const convo = [...data, userMsg].map((m) => ({ role: m.role, content: m.text }));
    startStreaming(convo, assistantId);
  };

  const Typing = () => (
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

  // Mic button: press to start, press again to stop+transcribe+send
  const MicButton = () => (
    <Pressable
      onPress={async () => {
        if (!rec.isRecording) {
          await rec.start();
        } else {
          const uri = await rec.stop();
          if (uri) {
            try {
              const { text } = await transcribeAudio(uri);
              if (text?.trim()) await send(text.trim());
            } catch (e) {
              console.warn("transcribe error", e);
            }
          }
        }
      }}
      style={{
        backgroundColor: rec.isRecording ? "#ef4444" : theme.colors.accent,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: theme.radius.pill,
      }}
    >
      <View style={{ width: 16, height: 16, borderRadius: 8, backgroundColor: "white" }} />
    </Pressable>
  );

  // Composer -> list bottom padding so messages never clash with the input row
  const composerHeight = inputHeight + ROW_VERT_PAD * 2;
  const listBottomPadding = composerHeight + theme.spacing.xl + insets.bottom;

  // --- SEND BUTTON STATE / STYLE ---
  const canSend = !!value.trim() && !sending && !typing;
  const isIdleDisabled = !value.trim() && !sending && !typing; // outlined when idle & empty

  return (
    <View style={{ flex: 1, backgroundColor: theme.colors.bg }}>
      <View style={{ height: insets.top, backgroundColor: theme.colors.surface }} />

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

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.select({ ios: "padding", android: undefined })}
        keyboardVerticalOffset={0}
      >
        <FlatList
          ref={listRef}
          data={data}
          keyExtractor={(i) => i.id}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={{
            paddingHorizontal: theme.spacing.lg,
            paddingTop: theme.spacing.lg,
            paddingBottom: listBottomPadding,
          }}
          renderItem={({ item }) => <Bubble msg={item} />}
          ListFooterComponent={typing ? <Typing /> : <View style={{ height: 0 }} />}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          extraData={[msgs, listBottomPadding, typing]}
        />

        <View
          style={{
            paddingHorizontal: theme.spacing.lg,
            paddingBottom: theme.spacing.lg + insets.bottom,
            backgroundColor: theme.colors.bg,
          }}
        >
          {/* INPUT ROW */}
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

            {/* Placeholder wrapper */}
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

            {/* SEND BUTTON - outlined when empty/idle, filled when actionable or busy */}
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
                  ? {
                      backgroundColor: "transparent",
                      borderWidth: 1,
                      borderColor: theme.colors.accent,
                    }
                  : {
                      backgroundColor: theme.colors.accent,
                      ...theme.shadow(4),
                    },
              ]}
            >
              {sending || typing ? (
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
