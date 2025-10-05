import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { Thread, Message } from '../types';

const KEY = 'mhc_threads_v1';

type ThreadsCtx = {
  threads: Thread[];
  loading: boolean;
  createThread: (title?: string) => Promise<Thread>;
  deleteThread: (id: string) => Promise<void>;
  appendMessage: (id: string, msg: Message) => Promise<void>;
  setTitle: (id: string, title: string) => Promise<void>;
  replaceMessages: (id: string, msgs: Message[]) => Promise<void>;
  reload: () => Promise<void>;
};

const Ctx = createContext<ThreadsCtx | null>(null);

// Persist helper (fire-and-forget)
async function persist(next: Thread[]) {
  try { await AsyncStorage.setItem(KEY, JSON.stringify(next)); } catch {}
}

export function ThreadsProvider({ children }: { children: React.ReactNode }) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const raw = await AsyncStorage.getItem(KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      setThreads(Array.isArray(parsed) ? parsed as Thread[] : []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void reload(); }, [reload]);

  // 🔒 atomic updater: never use stale `threads` closures
  const update = useCallback((fn: (prev: Thread[]) => Thread[]) => {
    setThreads(prev => {
      const next = fn(prev);
      persist(next);
      return next;
    });
  }, []);

  const createThread = useCallback(async (title = 'New Chat') => {
    const now = Date.now();
    const t: Thread = {
      id: Math.random().toString(36).slice(2),
      title,
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    // We need to return t, so mirror the change outside setState:
    let created = t;
    update(prev => [created, ...prev]);
    return created;
  }, [update]);

  const deleteThread = useCallback(async (id: string) => {
    update(prev => prev.filter(t => t.id !== id));
  }, [update]);

  const setTitle = useCallback(async (id: string, title: string) => {
    update(prev => prev.map(t => t.id === id ? { ...t, title, updatedAt: Date.now() } : t));
  }, [update]);

  const appendMessage = useCallback(async (id: string, msg: Message) => {
    update(prev => prev.map(t => {
      if (t.id !== id) return t;
      const msgs = [...t.messages, msg];
      return {
        ...t,
        messages: msgs,
        updatedAt: Date.now(),
        title:
          t.messages.length === 0 && msg.role === 'user'
            ? msg.text.slice(0, 32) + (msg.text.length > 32 ? '…' : '')
            : t.title,
      };
    }));
  }, [update]);

  const replaceMessages = useCallback(async (id: string, msgs: Message[]) => {
    update(prev => prev.map(t => t.id === id ? { ...t, messages: msgs, updatedAt: Date.now() } : t));
  }, [update]);

  return (
    <Ctx.Provider value={{ threads, loading, createThread, deleteThread, appendMessage, setTitle, replaceMessages, reload }}>
      {children}
    </Ctx.Provider>
  );
}

export function useThreads() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useThreads must be used within ThreadsProvider');
  return ctx;
}
