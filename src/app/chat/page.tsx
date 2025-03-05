import Chat from "@/components/chat/Chat";

export default function ChatPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-8">
      <div className="w-full max-w-5xl">
        <Chat />
      </div>
    </main>
  );
}
