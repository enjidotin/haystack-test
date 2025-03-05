"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { addDocument } from "@/lib/api";
import { toast } from "sonner";

const DocumentUpload: React.FC = () => {
  const [content, setContent] = React.useState("");
  const [isUploading, setIsUploading] = React.useState(false);
  const [open, setOpen] = React.useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!content.trim()) {
      toast.error("Please enter document content");
      return;
    }

    setIsUploading(true);

    try {
      await addDocument(content);
      toast.success("Document added successfully");
      setContent("");
      setOpen(false);
    } catch (error) {
      console.error("Error adding document:", error);
      toast.error("Failed to add document");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          Add Document
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Add Document for RAG</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Textarea
            placeholder="Enter document content..."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="min-h-[200px]"
            disabled={isUploading}
          />
          <div className="flex justify-end">
            <Button type="submit" disabled={isUploading || !content.trim()}>
              {isUploading ? "Adding..." : "Add Document"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default DocumentUpload;
