// src/components/Bubble.tsx
import React from 'react';
import { View, Text } from 'react-native';
import dayjs from 'dayjs';
import { theme } from '../ui/theme';
import type { Message } from '../types';

export default function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user';
  const borderColor = isUser
    ? (theme.colors as any).bubbleUserBorder ?? theme.colors.accent
    : (theme.colors as any).bubbleAIBorder ?? theme.colors.line;

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
        borderColor,
        ...theme.shadow(1), // subtle lift for readability
      }}
    >
      <Text
        style={{
          color: theme.colors.text,
          fontSize: theme.type.body,
          lineHeight: theme.type.lineBody,
        }}
      >
        {msg.text}
      </Text>
      <Text
        style={{
          color: theme.colors.subtext,
          fontSize: theme.type.micro,
          marginTop: 6,
          alignSelf: 'flex-end',
        }}
      >
        {dayjs(msg.createdAt).format('HH:mm')}
      </Text>
    </View>
  );
}
