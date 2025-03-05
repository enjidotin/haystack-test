# AI Chat Application with Haystack and ChromaDB

A fullstack AI chat application built with Next.js, Haystack, and ChromaDB. This application allows users to chat with AI models and supports Retrieval Augmented Generation (RAG) for more contextual responses.

## Features

- 🤖 Chat with multiple AI models:
  - OpenAI models (GPT-3.5, GPT-4o, GPT-4 Turbo)
  - Google Gemini 2.0 Pro
- 🔄 Model switching between different AI models
- 📚 RAG (Retrieval Augmented Generation) support
- 📄 PDF document upload and indexing
- 💬 Question answering based on uploaded documents
- 💾 Document storage with ChromaDB
- 🎨 Modern UI with Tailwind CSS and shadcn/ui
- 🚀 Fast and responsive design

## Tech Stack

### Frontend

- Next.js 14
- TypeScript
- Tailwind CSS
- shadcn/ui components
- React Hooks

### Backend

- Python FastAPI
- Haystack AI framework
- ChromaDB for vector storage
- OpenAI API integration
- Google Generative AI for Gemini

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- OpenAI API key
- Google API key (for Gemini)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-chat-app.git
   cd ai-chat-app
   ```

2. Install frontend dependencies:

   ```bash
   npm install
   ```

3. Install backend dependencies:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Create environment variables:
   - Copy `.env.example` to `.env` in the backend directory
   - Add your OpenAI API key to the `.env` file
   - Add your Google API key to the `.env` file (for Gemini)

### Running the Application

1. Start the backend server:

   ```bash
   cd backend
   python main.py
   ```

2. In a new terminal, start the frontend:

   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Type a message in the chat input and press Enter or click the send button
2. Switch between different AI models using the model selector
3. Toggle RAG mode to enable context-aware responses
4. Upload PDF documents to ask questions about their content
5. Add text documents to enhance RAG capabilities

### PDF Indexing and Q&A

The application allows users to:

1. Upload PDF documents via the PDF Uploader
2. The system automatically extracts text from the PDF and indexes it
3. Enable RAG mode to ask questions about the uploaded PDFs
4. The system retrieves relevant chunks from the PDFs to answer your questions

### Google Gemini Integration

The application supports Google's Gemini 2.0 Pro model:

1. Select "gemini-pro" from the model dropdown
2. Use it for regular chat or in RAG mode
3. Gemini works seamlessly with the PDF Q&A functionality

## Project Structure

```
ai-chat-app/
├── src/                  # Frontend source code
│   ├── app/              # Next.js app directory
│   ├── components/       # React components
│   │   └── chat/         # Chat-related components
│   ├── lib/              # Utility functions and API clients
│   └── types/            # TypeScript type definitions
├── backend/              # Python backend
│   ├── main.py           # FastAPI application
│   └── requirements.txt  # Python dependencies
└── public/               # Static assets
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Haystack](https://github.com/deepset-ai/haystack) - For the AI framework
- [ChromaDB](https://github.com/chroma-core/chroma) - For the vector database
- [shadcn/ui](https://ui.shadcn.com/) - For the beautiful UI components
- [Next.js](https://nextjs.org/) - For the frontend framework
- [Google Generative AI](https://ai.google.dev/) - For Gemini integration
