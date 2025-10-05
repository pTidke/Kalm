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
} from "react-native";
import { RouteProp, useRoute } from "@react-navigation/native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
dayjs.extend(relativeTime);

import { useThreads } from "../store/threads";
import type { Message, RootStackParamList } from "../types";
import { sendChat, transcribeAudio } from "../services/api";
import Bubble from "../components/Bubble";
import { theme } from "../ui/theme";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

type ChatRoute = RouteProp<RootStackParamList, "Chat">;

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

  // Dynamically sized input height (1 line baseline = 22)
  const [inputHeight, setInputHeight] = useState(22);
  const INPUT_ROW_VERT_PAD = 8; // must match paddingVertical on the row

  // Recorder
  const rec = useAudioRecorder();

  // Local optimistic list
  const [msgs, setMsgs] = useState<Message[]>([]);

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
      <View
        style={{
          flex: 1,
          backgroundColor: theme.colors.bg,
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <ActivityIndicator />
        <Text style={{ color: theme.colors.subtext, marginTop: 12 }}>
          Loading chat…
        </Text>
      </View>
    );
  }

  const send = async (overrideText?: string) => {
    const text = (overrideText ?? value).trim();
    if (!text || sending) return;

    const userMsg: Message = {
      id: Math.random().toString(),
      role: "user",
      text,
      createdAt: Date.now(),
    };

    setMsgs((prev) => [...prev, userMsg]);
    scrollToEnd();
    appendMessage(thread.id, userMsg).catch(() => {});
    if (!overrideText) setValue("");

    setSending(true);
    setTyping(true);

    try {
      const convo = [...data, userMsg].map((m) => ({
        role: m.role,
        content: m.text,
      }));
      const { reply, intent } = await sendChat(convo);

      const botMsg: Message = {
        id: Math.random().toString(),
        role: "assistant",
        text: reply,
        createdAt: Date.now(),
        intent,
      };
      setMsgs((prev) => [...prev, botMsg]);
      appendMessage(thread.id, botMsg).catch(() => {});
    } catch {
      const err: Message = {
        id: Math.random().toString(),
        role: "assistant",
        text: "Sorry—something went wrong. Please try again.",
        createdAt: Date.now(),
      };
      setMsgs((prev) => [...prev, err]);
      appendMessage(thread.id, err).catch(() => {});
    } finally {
      setSending(false);
      setTyping(false);
      setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 50);
    }
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
      <Text style={{ color: theme.colors.subtext }}>…</Text>
    </View>
  );

  // Mic: tap to start, tap to stop → transcribe → send
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
        backgroundColor: rec.isRecording ? "#ef4444" : "#0ea5e9",
        paddingVertical: 10,
        paddingHorizontal: 14,
        borderRadius: theme.radius.pill,
      }}
    >
      <View
        style={{
          width: 18,
          height: 18,
          borderRadius: 9,
          backgroundColor: "white",
        }}
      />
    </Pressable>
  );

  // --- Dynamic bottom padding so last message never touches the composer ---
  const composerHeight = inputHeight + INPUT_ROW_VERT_PAD * 2; // row padding + input
  const listBottomPadding =
    composerHeight + theme.spacing.xl + insets.bottom; // extra cushion

  return (
    <View style={{ flex: 1, backgroundColor: theme.colors.bg }}>
      <View
        style={{ height: insets.top, backgroundColor: theme.colors.surface }}
      />

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
        <Text
          style={{ color: theme.colors.text, fontSize: 20, fontWeight: "800" }}
        >
          {thread.title}
        </Text>
        <Text
          style={{ color: theme.colors.subtext, fontSize: 12, marginTop: 6 }}
        >
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
            // 👇 dynamic bottom padding = composer height + safe area + cushion
            paddingBottom: listBottomPadding,
          }}
          renderItem={({ item }) => <Bubble msg={item} />}
          ListFooterComponent={typing ? <Typing /> : <View style={{ height: 0 }} />}
          onContentSizeChange={() =>
            listRef.current?.scrollToEnd({ animated: true })
          }
          extraData={[msgs, listBottomPadding]} // re-render when height changes
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
              paddingVertical: INPUT_ROW_VERT_PAD,
              ...theme.shadow(6),
            }}
          >
            <MicButton />

            {/* Placeholder wrapper */}
            <View style={{ flex: 1, justifyContent: "center" }}>
              {!value && !focused && (
                <View
                  pointerEvents="none"
                  style={{
                    position: "absolute",
                    left: 0,
                    right: 0,
                    top: 0,
                    bottom: 0,
                    justifyContent: "center",
                  }}
                >
                  <Text
                    numberOfLines={1}
                    style={{ color: theme.colors.subtext, fontSize: 16 }}
                  >
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
                  const h = Math.ceil(e.nativeEvent.contentSize.height || 22);
                  setInputHeight(Math.max(22, Math.min(120, h)));
                }}
                style={{
                  height: inputHeight, // explicit height keeps vertical center
                  color: theme.colors.text,
                  fontSize: 16,
                  lineHeight: 22,
                  paddingTop: 0,
                  paddingBottom: 0,
                  margin: 0,
                  textAlignVertical: "center", // Android
                  // @ts-ignore Android-only
                  includeFontPadding: false,
                }}
              />
            </View>

            <Pressable
              onPress={() => send()}
              disabled={!value.trim() || sending}
              style={{
                backgroundColor:
                  !value.trim() || sending ? "#334155" : theme.colors.accent,
                paddingVertical: 10,
                paddingHorizontal: 16,
                borderRadius: theme.radius.pill,
                ...theme.shadow(4),
              }}
            >
              {sending ? (
                <ActivityIndicator color="#0b1220" />
              ) : (
                <Text style={{ color: "#0b1220", fontWeight: "800" }}>Send</Text>
              )}
            </Pressable>
          </View>

          {rec.error ? (
            <Text
              style={{
                color: "#fca5a5",
                fontSize: 12,
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
              fontSize: 11,
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
