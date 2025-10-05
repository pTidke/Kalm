import React from 'react';
import { View, Text } from 'react-native';
import dayjs from 'dayjs';
import { theme } from '../ui/theme';
import type { Message } from '../types';

export default function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user';
  return (
    <View
      style={{
        alignSelf: isUser ? 'flex-end' : 'flex-start',
        backgroundColor: isUser ? theme.colors.bubbleUser : theme.colors.bubbleAI,
        paddingVertical: 10,
        paddingHorizontal: 14,
        borderRadius: theme.radius.xl,
        marginVertical: 6,
        maxWidth: '86%',
        borderWidth: 1,
        borderColor: isUser ? '#2aa9c0' : theme.colors.line, // brighter border for user
      }}
    >
      <Text style={{ color: theme.colors.text, fontSize: 16, lineHeight: 22 }}>
        {msg.text}
      </Text>
      <Text style={{ color: theme.colors.subtext, fontSize: 11, marginTop: 6, alignSelf: 'flex-end' }}>
        {dayjs(msg.createdAt).format('HH:mm')}
      </Text>
    </View>
  );
}
