import React from 'react';
import { Pressable, Text, View } from 'react-native';

export function MicButton({
  active,
  onPress,
}: {
  active: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={{
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 999,
        backgroundColor: active ? '#ef4444' : '#0ea5e9',
      }}
    >
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
        <View
          style={{
            width: 18,
            height: 18,
            borderRadius: 9,
            backgroundColor: 'white',
          }}
        />
        <Text style={{ color: 'white', fontWeight: '600' }}>
          {active ? 'Stop' : 'Speak'}
        </Text>
      </View>
    </Pressable>
  );
}
