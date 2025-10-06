// src/ui/theme.ts
import { Platform } from "react-native";

// Calm LIGHT palette — soft blues/greens, dark text on light bg.
const palette = {
  bg: "#F7FAFC",
  surface: "#FFFFFF",
  line: "#E6EEF3",
  text: "#0E293B",
  subtext: "#5E6B75",
  accent: "#1FB6FF",

  bubbleUser: "#EAF7FF",
  bubbleAI: "#FFFFFF",

  // NEW — soft borders so bubbles don’t blend in
  bubbleUserBorder: "#BFE9FF", // pale accent
  bubbleAIBorder: "#E6EEF3",   // light gray-blue
};

const spacing = {
  xs: 6,
  sm: 10,
  md: 14,
  lg: 18,
  xl: 22,
};

const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 22,
  pill: 999,
};

// ↓↓↓ GLOBAL FONT SCALE (make smaller by lowering this)
export const fontScale = 0.88;

// convenient helpers
export const fs = (n: number) => Math.max(10, Math.round(n * fontScale));
export const lh = (n: number, ratio = 1.25) => Math.round(fs(n) * ratio);

// Softer shadow for light mode
const shadow = (elevation = 4) =>
  Platform.select({
    ios: {
      shadowColor: "#000",
      shadowOpacity: 0.08,
      shadowRadius: 12,
      shadowOffset: { width: 0, height: Math.ceil(elevation / 2) },
    },
    android: { elevation },
    default: {},
  })!;

export const theme = {
  colors: palette,
  spacing,
  radius,
  shadow,
  type: {
    title: fs(18),
    body: fs(15),
    small: fs(11),
    micro: fs(10),
    lineBody: lh(15),
  },
};
