// src/components/TypingDots.tsx
import React, { useEffect, useRef } from "react";
import { Animated, Text, View } from "react-native";
import { theme } from "../ui/theme";

export default function TypingDots() {
  const dot = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(dot, { toValue: 1, duration: 400, useNativeDriver: true }),
        Animated.timing(dot, { toValue: 0, duration: 400, useNativeDriver: true }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [dot]);

  const opacity = dot.interpolate({ inputRange: [0, 1], outputRange: [0.2, 1] });

  return (
    <View
      style={{
        alignSelf: "flex-start",
        backgroundColor: theme.colors.bubbleAI,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: theme.radius.lg,
        marginVertical: 6,
        borderWidth: 1,
        borderColor: theme.colors.line,
      }}
    >
      <Animated.Text style={{ opacity, color: theme.colors.subtext }}>typing…</Animated.Text>
    </View>
  );
}
