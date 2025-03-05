"use client";

import React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ChatMessage } from "@/types";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ChatMessagesProps {
  messages: ChatMessage[];
  loading?: boolean;
}

const ChatMessages: React.FC<ChatMessagesProps> = ({
  messages,
  loading = false,
}) => {
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
        <h3 className="text-lg font-semibold mb-2">Welcome to AI Chat</h3>
        <p className="text-sm text-muted-foreground">
          Start a conversation by typing a message below.
        </p>
      </div>
    );
  }

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="flex flex-col gap-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div className="flex gap-3 max-w-[80%]">
              {message.role !== "user" && (
                <Avatar className="h-8 w-8">
                  <AvatarFallback>AI</AvatarFallback>
                  <AvatarImage src="/bot-avatar.png" />
                </Avatar>
              )}
              <Card
                className={`p-3 ${
                  message.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : ""
                }`}
              >
                <div className="whitespace-pre-wrap">{message.content}</div>
              </Card>
              {message.role === "user" && (
                <Avatar className="h-8 w-8">
                  <AvatarFallback>ME</AvatarFallback>
                  <AvatarImage src="/user-avatar.png" />
                </Avatar>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[80%]">
              <Avatar className="h-8 w-8">
                <AvatarFallback>AI</AvatarFallback>
                <AvatarImage src="/bot-avatar.png" />
              </Avatar>
              <Card className="p-3">
                <div className="flex gap-1">
                  <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce"></div>
                  <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce delay-75"></div>
                  <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce delay-150"></div>
                </div>
              </Card>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
};

export default ChatMessages;
