export type Role = 'user' | 'assistant';

export type Message = {
  id: string;
  role: Role;
  text: string;
  createdAt: number;
  intent?: string;
};

export type Thread = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
};

export type RootStackParamList = {
  List: undefined;
  Chat: { id: string };
};
