import fs from "fs";
import path from "path";
import { Metadata } from "next";
import { markdownToHtml } from "@/lib/markdown";
import "./markdown.css";

export const metadata: Metadata = {
  title: "Quick Start Guide - AI Chat Application",
  description:
    "Get started with the PDF indexing and Google Gemini features of the AI Chat Application",
};

async function getMarkdownContent() {
  const filePath = path.join(process.cwd(), "QUICK-START.md");
  try {
    const fileContents = fs.readFileSync(filePath, "utf8");
    return fileContents;
  } catch (error) {
    console.error("Error reading QUICK-START.md:", error);
    return "# Quick Start Guide Not Found\n\nSorry, the quick start guide could not be found.";
  }
}

export default async function QuickStartPage() {
  const markdownContent = await getMarkdownContent();
  const htmlContent = await markdownToHtml(markdownContent);

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-4 md:p-8 pt-12">
      <div className="w-full max-w-4xl prose dark:prose-invert prose-headings:scroll-mt-20">
        <div dangerouslySetInnerHTML={{ __html: htmlContent }} />
        <div className="mt-16 flex justify-center">
          <a
            href="/chat"
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Try the Chat App
          </a>
        </div>
      </div>
    </main>
  );
}
