import { ChatMessage } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatRequest {
  messages: ChatMessage[];
  model: string;
}

export interface ChatResponse {
  response: string;
  model: string;
}

export const fetchModels = async (): Promise<string[]> => {
  try {
    const response = await fetch(`${API_URL}/models`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.models;
  } catch (error) {
    console.error("Error fetching models:", error);
    return ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]; // Fallback models
  }
};

export const sendMessage = async (
  request: ChatRequest
): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.detail || `HTTP error! status: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error sending message:", error);
    throw error;
  }
};

export const sendMessageWithRAG = async (
  request: ChatRequest
): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${API_URL}/chat/rag`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.detail || `HTTP error! status: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error sending RAG message:", error);
    throw error;
  }
};

export const addDocument = async (
  content: string,
  meta: Record<string, string | number | boolean> = {}
): Promise<{ document_id: string }> => {
  try {
    const response = await fetch(`${API_URL}/documents`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ content, meta }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.detail || `HTTP error! status: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error adding document:", error);
    throw error;
  }
};
