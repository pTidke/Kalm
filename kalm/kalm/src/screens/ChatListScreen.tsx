import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  FlatList,
  Pressable,
  TextInput,
  Alert,
  ScrollView,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import dayjs from 'dayjs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useThreads } from '../store/threads';
import type { Thread, RootStackParamList, Message } from '../types';
import { theme, fs } from '../ui/theme';

// ---- Types ----
type Nav = NativeStackNavigationProp<RootStackParamList, 'Chat'>;

type FocusTemplate = {
  key: string;
  chip: string;       // visible chip text
  seedUser: string;   // first user message
  systemHint: string; // hidden system background prompt
};

const FOCUS_TEMPLATES: FocusTemplate[] = [
  {
    key: 'anxiety',
    chip: 'Anxiety check-in 🌿',
    seedUser: "I'm feeling quite anxious today and it's hard to relax.",
    systemHint:
      'Focus on grounding, paced breathing (inhale 4 / exhale 6), and normalizing anxiety. Be concise, warm, and ask one gentle follow-up.',
  },
  {
    key: 'sleep',
    chip: 'Sleep journal 😴',
    seedUser: "My sleep has been rough and I can’t switch off my mind at night.",
    systemHint:
      'Use CBT-I tips (sleep window, wind-down, stimulus control) in plain language. Offer one small step tonight.',
  },
  {
    key: 'stress',
    chip: 'Stress debrief 🧭',
    seedUser: "I’m overwhelmed with work and can’t keep up.",
    systemHint:
      'Help with prioritization, break tasks into micro-steps, propose a 2-minute starter action. Keep it validating.',
  },
  {
    key: 'mood',
    chip: 'Low mood support 💛',
    seedUser: "My mood’s been low and I’m not enjoying things lately.",
    systemHint:
      'Behavioral activation ideas + gentle validation. Suggest one tiny meaningful action today; ask a kind follow-up.',
  },
];

const pastelBgs = ['#E8F3FF', '#EAF7F0', '#FFF3EC', '#F5F0FF', '#F8F9E9'];

function hashToIdx(s: string | number, mod: number) {
  const str = String(s);
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
  return Math.abs(h) % mod;
}

function emojiForThread(t: Thread) {
  const last = t.messages?.[t.messages.length - 1];
  const intent = (last as any)?.intent as string | undefined;
  if (intent) {
    if (intent.includes('anxiety')) return '🌿';
    if (intent.includes('sleep')) return '😴';
    if (intent.includes('stress')) return '🧭';
    if (intent.includes('low_mood')) return '💛';
    if (intent.includes('relationship')) return '💬';
  }
  const name = (t.title || '').toLowerCase();
  if (name.includes('sleep')) return '😴';
  if (name.includes('stress')) return '🧭';
  if (name.includes('anx')) return '🌿';
  if (name.includes('mood') || name.includes('low')) return '💛';
  return '💬';
}

function Avatar({ t }: { t: Thread }) {
  const idx = hashToIdx(t.id, pastelBgs.length);
  const bg = pastelBgs[idx];
  return (
    <View
      style={{
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: bg,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: theme.colors.line,
      }}
    >
      <Text style={{ fontSize: fs(18) }}>{emojiForThread(t)}</Text>
    </View>
  );
}

function Chevron() {
  return (
    <Text
      style={{
        color: theme.colors.subtext,
        fontSize: fs(16),
        marginLeft: theme.spacing.sm,
      }}
    >
      ›
    </Text>
  );
}

function ThreadCard({
  t,
  onPress,
  onLongPress,
}: {
  t: Thread;
  onPress: () => void;
  onLongPress: () => void;
}) {
  const last = t.messages?.[t.messages.length - 1];
  return (
    <Pressable
      onPress={onPress}
      onLongPress={onLongPress}
      style={{
        backgroundColor: theme.colors.surface,
        borderRadius: theme.radius.lg,
        borderWidth: 1,
        borderColor: theme.colors.line,
        paddingVertical: 12,
        paddingHorizontal: 14,
        ...theme.shadow(2),
      }}
    >
      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        <Avatar t={t} />
        <View style={{ flex: 1, marginLeft: theme.spacing.md }}>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Text
              numberOfLines={1}
              style={{
                flex: 1,
                color: theme.colors.text,
                fontSize: theme.type.title,
                fontWeight: '800',
              }}
            >
              {t.title || 'Untitled chat'}
            </Text>
            <Text
              style={{
                marginLeft: theme.spacing.sm,
                color: theme.colors.subtext,
                fontSize: theme.type.micro,
              }}
            >
              {dayjs(t.updatedAt).format('MMM D • HH:mm')}
            </Text>
            <Chevron />
          </View>

          <Text
            numberOfLines={2}
            style={{
              marginTop: 4,
              color: theme.colors.subtext,
              fontSize: theme.type.body,
              lineHeight: theme.type.lineBody,
            }}
          >
            {last?.text?.trim() || 'No messages yet'}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

function SearchBar({
  value,
  onChange,
}: {
  value: string;
  onChange: (s: string) => void;
}) {
  return (
    <View
      style={{
        backgroundColor: theme.colors.surface,
        borderRadius: theme.radius.xl,
        borderWidth: 1,
        borderColor: theme.colors.line,
        paddingHorizontal: theme.spacing.lg,
        paddingVertical: 10,
        flexDirection: 'row',
        alignItems: 'center',
        ...theme.shadow(1),
      }}
    >
      <Text style={{ color: theme.colors.subtext, marginRight: 8 }}>🔎</Text>
      <TextInput
        placeholder="Search chats…"
        placeholderTextColor={theme.colors.subtext}
        value={value}
        onChangeText={onChange}
        style={{
          flex: 1,
          color: theme.colors.text,
          fontSize: theme.type.body,
          padding: 0,
          margin: 0,
        }}
      />
    </View>
  );
}

function QuickChips({ onPick }: { onPick: (tpl: FocusTemplate) => void }) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={{ gap: 8 }}
    >
      {FOCUS_TEMPLATES.map((c) => (
        <Pressable
          key={c.key}
          onPress={() => onPick(c)}
          style={{
            borderRadius: theme.radius.pill,
            borderWidth: 1,
            borderColor: theme.colors.accent,
            backgroundColor: theme.colors.surface,
            paddingVertical: 8,
            paddingHorizontal: 12,
          }}
        >
          <Text
            style={{
              color: theme.colors.accent,
              fontSize: theme.type.small,
              fontWeight: '700',
            }}
          >
            {c.chip}
          </Text>
        </Pressable>
      ))}
    </ScrollView>
  );
}

export default function ChatListScreen() {
  const insets = useSafeAreaInsets();
  const nav = useNavigation<Nav>();
  const { threads, createThread, deleteThread, appendMessage } = useThreads();
  const [query, setQuery] = useState('');

  const open = (t: Thread, seed?: string) => nav.navigate('Chat', { id: t.id, seed });

  const onNew = async (title?: string) => {
    const t = await createThread(title);
    open(t);
  };

  const onDelete = (t: Thread) => {
    Alert.alert('Delete chat?', `“${t.title}” will be removed.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => deleteThread(t.id) },
    ]);
  };

  // When a focused chip is tapped: create thread, save hidden system prompt, navigate with seed
  const onPickTemplate = async (tpl: FocusTemplate) => {
    const t = await createThread(tpl.chip);
    const sys: Message = {
      id: Math.random().toString(),
      role: 'system',
      text: tpl.systemHint,
      createdAt: Date.now(),
    };
    await appendMessage(t.id, sys);
    open(t, tpl.seedUser);
  };

  const data = useMemo(() => {
    const list = [...threads].sort((a, b) => b.updatedAt - a.updatedAt);
    if (!query.trim()) return list;
    const q = query.toLowerCase();
    return list.filter(
      (t) =>
        t.title.toLowerCase().includes(q) ||
        t.messages?.some((m) => m.text?.toLowerCase().includes(q))
    );
  }, [threads, query]);

  return (
    <View style={{ flex: 1, backgroundColor: theme.colors.bg }}>
      {/* Safe area top on surface */}
      <View style={{ height: insets.top, backgroundColor: theme.colors.surface }} />

      {/* Header */}
      <View
        style={{
          paddingHorizontal: theme.spacing.xl,
          paddingTop: theme.spacing.lg,
          paddingBottom: theme.spacing.sm,
          borderBottomWidth: 1,
          borderBottomColor: theme.colors.line,
          backgroundColor: theme.colors.surface,
        }}
      >
        <Text
          style={{
            color: theme.colors.text,
            fontSize: theme.type.title + 2,
            fontWeight: '800',
          }}
        >
          Kalm Companion
        </Text>
      </View>

      <FlatList
        data={data}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={{
          paddingHorizontal: theme.spacing.lg,
          paddingTop: theme.spacing.lg,
          paddingBottom: theme.spacing.xl + insets.bottom + 72, // room for FAB
          gap: 10,
        }}
        ListHeaderComponent={
          <View style={{ marginBottom: theme.spacing.lg }}>
            {/* Hero card */}
            <View
              style={{
                backgroundColor: theme.colors.surface,
                borderRadius: theme.radius.lg,
                borderWidth: 1,
                borderColor: theme.colors.line,
                padding: 16,
                marginBottom: theme.spacing.md,
                ...theme.shadow(2),
              }}
            >
              <Text
                style={{
                  color: theme.colors.text,
                  fontSize: theme.type.title,
                  fontWeight: '800',
                }}
              >
                How are you feeling today?
              </Text>
              <Text
                style={{
                  color: theme.colors.subtext,
                  fontSize: theme.type.body,
                  marginTop: 6,
                }}
              >
                Start a focused chat or search previous conversations.
              </Text>

              <View style={{ marginTop: 10 }}>
                <QuickChips onPick={onPickTemplate} />
              </View>
            </View>

            <SearchBar value={query} onChange={setQuery} />
          </View>
        }
        ListEmptyComponent={
          <View
            style={{
              backgroundColor: theme.colors.surface,
              borderRadius: theme.radius.lg,
              borderWidth: 1,
              borderColor: theme.colors.line,
              padding: 16,
              alignItems: 'center',
              ...theme.shadow(2),
            }}
          >
            <Text
              style={{
                color: theme.colors.text,
                fontSize: theme.type.title,
                fontWeight: '700',
              }}
            >
              Start your first chat
            </Text>
            <Text
              style={{
                color: theme.colors.subtext,
                fontSize: theme.type.body,
                marginTop: 6,
                textAlign: 'center',
              }}
            >
              Create a new conversation to begin. You can keep multiple chats for different topics.
            </Text>
            <Pressable
              onPress={() => onNew()}
              style={{
                marginTop: 12,
                backgroundColor: theme.colors.accent,
                paddingVertical: 10,
                paddingHorizontal: 16,
                borderRadius: theme.radius.pill,
                ...theme.shadow(4),
              }}
            >
              <Text style={{ color: '#0b1220', fontWeight: '800', fontSize: fs(14) }}>
                New Chat
              </Text>
            </Pressable>
          </View>
        }
        renderItem={({ item }) => (
          <ThreadCard
            t={item}
            onPress={() => open(item)}
            onLongPress={() => onDelete(item)}
          />
        )}
      />

      {/* FAB */}
      <Pressable
        onPress={() => onNew()}
        style={{
          position: 'absolute',
          right: theme.spacing.xl,
          bottom: theme.spacing.xl + insets.bottom,
          backgroundColor: theme.colors.accent,
          paddingVertical: 12,
          paddingHorizontal: 18,
          borderRadius: theme.radius.pill,
          borderWidth: 1,
          borderColor: 'rgba(0,0,0,0.06)',
          ...theme.shadow(6),
        }}
      >
        <Text style={{ color: '#0b1220', fontWeight: '800', fontSize: fs(14) }}>
          + New Chat
        </Text>
      </Pressable>
    </View>
  );
}
