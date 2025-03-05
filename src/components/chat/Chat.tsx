"use client";

import React from "react";
import { ChatMessage } from "@/types";
import { fetchModels, sendMessage, sendMessageWithRAG } from "@/lib/api";
import ChatMessages from "./ChatMessages";
import ChatInput from "./ChatInput";
import ModelSwitcher from "./ModelSwitcher";
import DocumentUpload from "./DocumentUpload";
import PDFUploader from "./PDFUploader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

const Chat: React.FC = () => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [models, setModels] = React.useState<string[]>([]);
  const [currentModel, setCurrentModel] =
    React.useState<string>("gpt-3.5-turbo");
  const [loading, setLoading] = React.useState<boolean>(false);
  const [useRAG, setUseRAG] = React.useState<boolean>(false);
  const [activeTab, setActiveTab] = React.useState<string>("chat");

  // Fetch available models on component mount
  React.useEffect(() => {
    const getModels = async () => {
      try {
        const availableModels = await fetchModels();
        setModels(availableModels);

        // Set default model from available models
        if (availableModels.length > 0) {
          setCurrentModel(availableModels[0]);
        }
      } catch (error) {
        console.error("Failed to fetch models:", error);
        // Set fallback models
        setModels(["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gemini-pro"]);
        toast.error("Failed to load available models");
      }
    };

    getModels();
  }, []);

  const handleSendMessage = async (content: string) => {
    // Add user message to chat
    const userMessage: ChatMessage = { role: "user", content };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      // Prepare all messages for context
      const allMessages = [...messages, userMessage];

      // Send to API - use RAG endpoint if RAG mode is enabled
      const response = useRAG
        ? await sendMessageWithRAG({
            messages: allMessages,
            model: currentModel,
          })
        : await sendMessage({
            messages: allMessages,
            model: currentModel,
          });

      // Add AI response to chat
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.response,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Failed to send message:", error);

      // Add error message
      const errorMessage: ChatMessage = {
        role: "assistant",
        content:
          "Sorry, I encountered an error processing your request. Please try again.",
      };

      setMessages((prev) => [...prev, errorMessage]);
      toast.error("Error processing your message");
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (model: string) => {
    setCurrentModel(model);
    toast.info(`Model switched to ${model}`);
  };

  const handleTabChange = (value: string) => {
    setActiveTab(value);
  };

  return (
    <Card className="w-full max-w-4xl mx-auto h-[90vh] flex flex-col">
      <CardHeader className="pb-2 border-b">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <CardTitle>AI Chat with Haystack & ChromaDB</CardTitle>

          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center space-x-2">
              <Switch
                id="rag-mode"
                checked={useRAG}
                onCheckedChange={setUseRAG}
              />
              <Label htmlFor="rag-mode">RAG Mode</Label>
            </div>

            {useRAG && (
              <div className="flex items-center gap-2">
                <DocumentUpload />
                <PDFUploader />
              </div>
            )}

            <ModelSwitcher
              models={models}
              currentModel={currentModel}
              onModelChange={handleModelChange}
              disabled={loading}
            />
          </div>
        </div>

        <Tabs
          defaultValue="chat"
          value={activeTab}
          onValueChange={handleTabChange}
          className="mt-2"
        >
          <TabsList>
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="about">About</TabsTrigger>
          </TabsList>
          <TabsContent value="about" className="text-sm p-2">
            <h3 className="font-semibold mb-1">Available Features:</h3>
            <ul className="list-disc pl-5 space-y-1">
              <li>Regular chat with OpenAI models (GPT-3.5, GPT-4o, etc.)</li>
              <li>Google Gemini 2.0 Pro integration</li>
              <li>RAG (Retrieval Augmented Generation) capabilities</li>
              <li>PDF document upload and indexing</li>
              <li>Text document upload for knowledge base</li>
            </ul>
            <p className="mt-2">
              Toggle <strong>RAG Mode</strong> to ask questions about uploaded
              documents. Upload documents via the <strong>Upload PDF</strong> or{" "}
              <strong>Add Document</strong> buttons.
            </p>
          </TabsContent>
        </Tabs>
      </CardHeader>

      <CardContent className="flex-1 p-0 flex flex-col">
        <ChatMessages messages={messages} loading={loading} />
        <ChatInput onSendMessage={handleSendMessage} disabled={loading} />
      </CardContent>
    </Card>
  );
};

export default Chat;
