export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatSession {
  id: string;
  messages: ChatMessage[];
  model: string;
  createdAt: Date;
  updatedAt: Date;
}

export type ModelType = "gpt-3.5-turbo" | "gpt-4o" | "gpt-4-turbo";
