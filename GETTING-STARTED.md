# Getting Started with the Enhanced AI Chat Application

This guide will help you get up and running with the AI Chat Application, including the newly added PDF indexing and Google Gemini features.

## Prerequisites

Before you begin, make sure you have:

1. **Node.js** (v18+) and **npm** installed
2. **Python** (v3.9+) installed
3. An **OpenAI API key** for using OpenAI models
4. A **Google API key** for using Gemini models (optional, but required for Gemini functionality)

## Step 1: Set Up Environment Variables

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Step 2: Install Dependencies

1. Install backend dependencies:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd ..
   npm install
   ```

## Step 3: Start the Application

1. Start the backend server:

   ```bash
   cd backend
   python main.py
   ```

2. In a new terminal window, start the frontend development server:

   ```bash
   npm run dev
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Step 4: Using the New Features

### PDF Indexing and Q&A

1. From the main page, click "Start Chatting" to open the chat interface
2. Click on the "PDF Uploader" tab at the bottom of the chat interface
3. Click "Upload PDF" and select a PDF file from your computer
4. After successful upload, switch back to the chat tab
5. Enable RAG mode using the toggle
6. Ask questions about the content of your uploaded PDF

### Google Gemini

1. In the chat interface, use the model switcher dropdown at the top
2. Select "gemini-pro" from the available models
3. You can use Gemini for both regular chat and RAG mode with PDFs

## Troubleshooting

If you encounter any issues:

1. **Backend errors**: Check the terminal where you're running the backend server for error messages
2. **Frontend errors**: Check your browser's developer console (F12) for error messages
3. **API key issues**: Verify that your API keys are correctly set in the `.env` file
4. **PDF upload problems**: Ensure your PDF is not too large and is a valid PDF format

## Additional Resources

- **README.md**: General information about the application
- **QUICK-START.md**: Detailed guide on using the new features
- **IMPLEMENTATION-SUMMARY.md**: Technical overview of the implementation

## Next Steps

Once you're comfortable with the basic features, you can:

1. Try uploading different types of PDFs to test the Q&A capabilities
2. Compare responses between OpenAI models and Google Gemini
3. Experiment with RAG mode to see how it enhances responses with context

Enjoy using your enhanced AI Chat Application!
