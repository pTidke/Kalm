import 'react-native-gesture-handler';
import React, { useEffect } from 'react';
import { View } from 'react-native';
import * as SplashScreen from 'expo-splash-screen';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { ThreadsProvider, useThreads } from './src/store/threads';
import ChatListScreen from './src/screens/ChatListScreen';
import ChatScreen from './src/screens/ChatScreen';
import type { RootStackParamList } from './src/types';

// keep splash on until we finish loading local threads
SplashScreen.preventAutoHideAsync().catch(() => {});

const Stack = createNativeStackNavigator<RootStackParamList>();

function Root() {
  const { loading } = useThreads();

  useEffect(() => {
    if (!loading) SplashScreen.hideAsync();
  }, [loading]);

  // While loading, render a blank view that matches splash bg to avoid flicker
  if (loading) {
    return <View style={{ flex: 1, backgroundColor: '#0b1220' }} />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator id={undefined} screenOptions={{ headerShown: false }}>
        <Stack.Screen name="List" component={ChatListScreen} />
        <Stack.Screen name="Chat" component={ChatScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <ThreadsProvider>
        <Root />
      </ThreadsProvider>
    </SafeAreaProvider>
  );
}
