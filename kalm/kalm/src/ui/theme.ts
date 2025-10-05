export const theme = {
  colors: {
    bg: '#0b1220',
    surface: '#0f172a',
    line: '#162036',
    text: '#e6eaf2',
    subtext: '#9fa6b2',
    bubbleAI: '#111827',
    bubbleUser: '#1b3a46',
    accent: '#22d3ee',
  },
  radius: { md: 12, lg: 18, xl: 24, pill: 9999 },
  spacing: { xs: 4, sm: 8, md: 12, lg: 16, xl: 24 },
  shadow(elevation = 6) {
    return {
      shadowColor: '#000',
      shadowOpacity: 0.2,
      shadowRadius: elevation,
      shadowOffset: { width: 0, height: Math.ceil(elevation / 2) },
      elevation,
    } as const;
  },
};
