import Link from "next/link";
import { Button } from "@/components/ui/button";
import { FileText, BookOpen, MessageSquare, PlayCircle } from "lucide-react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="relative flex place-items-center">
        <div className="flex flex-col items-center gap-6 text-center">
          <h1 className="text-4xl font-bold mb-2">AI Chat Application</h1>
          <p className="max-w-md text-xl text-muted-foreground">
            Chat with OpenAI and Google Gemini models, upload documents, and get
            AI-powered answers.
          </p>
          <div className="flex gap-4 mt-4">
            <Link href="/chat">
              <Button size="lg" className="gap-2">
                <MessageSquare className="h-5 w-5" />
                Start Chatting
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
