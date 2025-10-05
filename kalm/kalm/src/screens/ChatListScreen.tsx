import React from 'react';
import { View, Text, FlatList, Pressable, Alert } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import dayjs from 'dayjs';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useThreads } from '../store/threads';
import type { Thread, RootStackParamList } from '../types';

export default function ChatListScreen() {
  const insets = useSafeAreaInsets();
  const nav = useNavigation<NativeStackNavigationProp<RootStackParamList>>();

  const { threads, createThread, deleteThread } = useThreads();

  const open = (t: Thread) => nav.navigate('Chat', { id: t.id });

  const onNew = async () => {
    const t = await createThread();
    open(t);
  };

  const onDelete = (t: Thread) => {
    Alert.alert('Delete chat?', `“${t.title}” will be removed.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => deleteThread(t.id) }
    ]);
  };

  return (
    <View style={{
      flex: 1,
      backgroundColor: '#0b1220',
      paddingTop: insets.top,
      paddingBottom: insets.bottom
    }}>
      <View style={{ padding: 16, borderBottomWidth: 1, borderBottomColor: '#162036' }}>
        <Text style={{ color: '#e5e7eb', fontSize: 20, fontWeight: '700' }}>Calm Companion</Text>
        <Text style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>Choose a chat or start a new one</Text>
      </View>

      <FlatList
        data={[...threads].sort((a, b) => b.updatedAt - a.updatedAt)}
        keyExtractor={(item) => item.id}
        contentContainerStyle={{ padding: 12, gap: 8, paddingBottom: 12 + insets.bottom }}
        renderItem={({ item }) => (
          <Pressable
            onPress={() => open(item)}
            onLongPress={() => onDelete(item)}
            style={{ backgroundColor: '#0f172a', padding: 14, borderRadius: 12 }}
          >
            <Text style={{ color: '#e5e7eb', fontSize: 16, fontWeight: '600' }}>{item.title}</Text>
            <Text style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>
              {item.messages[item.messages.length - 1]?.text?.slice(0, 60) || 'No messages yet'}
            </Text>
            <Text style={{ color: '#64748b', fontSize: 11, marginTop: 6 }}>
              {dayjs(item.updatedAt).format('MMM D, HH:mm')}
            </Text>
          </Pressable>
        )}
        ListEmptyComponent={
          <View style={{ padding: 24 }}>
            <Text style={{ color: '#94a3b8' }}>No chats yet. Tap “New Chat”.</Text>
          </View>
        }
      />

      <Pressable
        onPress={onNew}
        style={{
          position: 'absolute',
          right: 20,
          bottom: 20 + insets.bottom,
          backgroundColor: '#22d3ee',
          paddingVertical: 14,
          paddingHorizontal: 18,
          borderRadius: 9999
        }}
      >
        <Text style={{ color: '#0f172a', fontWeight: '800' }}>New Chat</Text>
      </Pressable>
    </View>
  );
}
