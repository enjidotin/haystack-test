import os
import logging
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
import chromadb
import traceback
import uuid
import PyPDF2
import google.generativeai as genai
from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from haystack.dataclasses import Document
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.warning("Environment variables loaded")

# Log API key status (don't log the actual key)
api_key = os.environ.get("OPENAI_API_KEY", "")
if api_key:
    logger.warning("OpenAI API key found")
else:
    logger.warning("OpenAI API key not found in environment variables")

# Check for Google Gemini API key
gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
if gemini_api_key:
    logger.warning("Google Gemini API key found")
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
else:
    logger.warning("Google Gemini API key not found in environment variables")

# Create a Secret object for the OpenAI API key
api_key_secret = Secret.from_token(api_key)
logger.warning("API key Secret object created")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize document store
document_store = InMemoryDocumentStore()
logger.warning("Document store initialized")

# Initialize ChromaDB client for chat history and documents
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chat_collection = chroma_client.get_or_create_collection("chat_history")
    pdf_collection = chroma_client.get_or_create_collection("pdf_documents")
    logger.warning("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# Models available for selection
AVAILABLE_MODELS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
    "gemini-pro": "gemini-pro"  # Add Gemini Pro model
}
logger.warning(f"Available models: {list(AVAILABLE_MODELS.keys())}")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"

class ChatResponse(BaseModel):
    response: str
    model: str

class PDFResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    page_count: int = 0
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Chat API"}

@app.get("/models")
def get_models():
    logger.warning(f"Returning available models: {list(AVAILABLE_MODELS.keys())}")
    return {"models": list(AVAILABLE_MODELS.keys())}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.warning(f"Received chat request for model: {request.model}")
    if request.model not in AVAILABLE_MODELS:
        error_msg = f"Model {request.model} not available. Choose from {list(AVAILABLE_MODELS.keys())}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Create a basic chat pipeline
    try:
        # Format messages for the model
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response_text = ""
        
        # Check if using Gemini model
        if request.model == "gemini-pro":
            if not gemini_api_key:
                raise ValueError("Google Gemini API key not found in environment variables")
            
            try:
                # Use Google Gemini for generation
                logger.warning("Using Google Gemini model")

                gemini_model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp")
                
                # Convert to Gemini's format
                gemini_messages = []
                for msg in request.messages:
                    role = "user" if msg.role == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [msg.content]})
                
                # Generate response
                chat = gemini_model.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(gemini_messages[-1]["parts"][0])
                response_text = response.text
                logger.warning("Gemini model completed successfully")
            except Exception as e:
                logger.error(f"Error using Gemini model: {str(e)}")
                raise
        else:
            # Using OpenAI models
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            # Initialize the OpenAI generator with the selected model
            logger.warning(f"Initializing OpenAI generator with model: {AVAILABLE_MODELS[request.model]}")
            generator = OpenAIGenerator(
                api_key=api_key_secret,
                model=AVAILABLE_MODELS[request.model]
            )
            
            # Run generation
            logger.warning("Running generator...")
            
            try:
                # First, try passing the formatted messages as generation_kwargs
                result = generator.run(generation_kwargs={"messages": formatted_messages})
                logger.warning("Generator completed successfully using generation_kwargs")
                response_text = result["replies"][0]
            except Exception as first_error:
                logger.warning(f"First generator approach failed: {str(first_error)}")
                try:
                    # Second approach: try with prompt parameter for single messages
                    # Extract the last user message as prompt
                    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
                    if not last_user_message:
                        raise ValueError("No user message found")
                        
                    result = generator.run(prompt=last_user_message)
                    logger.warning("Generator completed successfully using prompt parameter")
                    response_text = result["replies"][0]
                except Exception as second_error:
                    logger.error(f"Second generator approach failed: {str(second_error)}")
                    raise ValueError(f"Could not run generator. First error: {str(first_error)}, Second error: {str(second_error)}")
        
        # Save to ChromaDB for history (optional)
        user_message = next((msg.content for msg in request.messages if msg.role == "user"), "")
        if user_message:
            try:
                chat_collection.add(
                    documents=[response_text],
                    metadatas=[{"role": "assistant", "model": request.model}],
                    ids=[f"response_{len(chat_collection.get()['ids']) + 1}"]
                )
                logger.warning("Response saved to ChromaDB")
            except Exception as e:
                logger.error(f"Error saving to ChromaDB (non-critical): {e}")
        
        response_data = {
            "response": response_text,
            "model": request.model
        }
        logger.warning(f"Returning response with length: {len(response_text)}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/rag")
async def chat_with_rag(request: ChatRequest):
    """
    Chat with Retrieval Augmented Generation (RAG) using documents from the store
    """
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available. Choose from {list(AVAILABLE_MODELS.keys())}")
    
    try:
        # Get the user's query (last user message)
        user_query = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        
        if not user_query:
            raise HTTPException(status_code=400, detail="No user query found in messages")
        
        # Retrieve relevant document chunks
        try:
            # Use ChromaDB to get relevant PDF chunks
            results = pdf_collection.query(
                query_texts=[user_query],
                n_results=5
            )
            
            logger.warning("results: " + str(results))
            
            # Extract chunks from results
            context_chunks = results.get('documents', [[]])[0]
            context = "\n\n".join(context_chunks)
            logger.warning(f"Retrieved {len(context_chunks)} document chunks for RAG")
            logger.warning(context)
            if not context_chunks:
                logger.warning("No document chunks found for RAG, falling back to regular chat")
                return await chat(request)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.warning("Error in document retrieval, falling back to regular chat")
            return await chat(request)
        
        # Check which model to use for generation
        if request.model == "gemini-pro":
            if not gemini_api_key:
                raise ValueError("Google Gemini API key not found in environment variables")
            
            try:
                # Use Google Gemini for RAG generation
                logger.warning("Using Google Gemini model for RAG")
                gemini_model = genai.GenerativeModel(
                    model_name="gemini-2.0-pro-exp"  # Correct model name
                )
                # Create a RAG prompt with the retrieved context
                rag_prompt = f"""
                Answer the following question based on the provided context. 
                If the answer is not in the context, say that you don't know.
                
                Context:
                {context}
                
                Question: {user_query}
                
                Answer:
                """
                
                # Generate response
                response = gemini_model.generate_content(rag_prompt)
                response_text = response.text
                logger.warning("Gemini RAG completed successfully")
                
                return {
                    "response": response_text,
                    "model": request.model
                }
            except Exception as e:
                logger.error(f"Error using Gemini model for RAG: {str(e)}")
                raise
        else:
            # Check if we have a valid OpenAI API key
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            # Create RAG pipeline
            
            # Prepare prompt with manually retrieved chunks
            prompt_template = """
            Answer the following question based on the given context:
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            # Replace template variables manually
            prompt = prompt_template.replace("{context}", context).replace("{query}", user_query)
            
            # Add generator
            generator = OpenAIGenerator(
                api_key=api_key_secret,
                model=AVAILABLE_MODELS[request.model]
            )
            
            try:
                # Try direct generation with the prompt
                result = generator.run(prompt=prompt)
                logger.warning("Generator completed successfully with direct prompt")
                response_text = result["replies"][0]
            except Exception as e:
                logger.error(f"Error in direct generation: {str(e)}")
                raise
            
            return {
                "response": response_text,
                "model": request.model
            }
    
    except Exception as e:
        logger.error(f"Error in RAG endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf", response_model=PDFResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file for RAG
    """
    if not file.filename.lower().endswith('.pdf'):
        return PDFResponse(success=False, message="Only PDF files are supported")
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        
        # Save uploaded file to temp location
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            return PDFResponse(success=False, message=f"Error processing file: {str(e)}")
        
        # Process PDF
        try:
            pdf_reader = PyPDF2.PdfReader(temp_file.name)
            page_count = len(pdf_reader.pages)
            logger.warning(f"Processing PDF with {page_count} pages")
            
            # Extract text from each page and create chunks
            chunks = []
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Split page into smaller chunks if needed
                    # Here we split by paragraphs, but you can use more sophisticated methods
                    paragraphs = text.split('\n\n')
                    for j, para in enumerate(paragraphs):
                        if para.strip():
                            chunks.append({
                                "text": para.strip(),
                                "metadata": {
                                    "source": file.filename,
                                    "page": i + 1,
                                    "chunk": j + 1
                                }
                            })
            
            logger.warning(f"Created {len(chunks)} text chunks from PDF")
            
            # Add chunks to ChromaDB
            document_id = str(uuid.uuid4())
            
            # Prepare data for ChromaDB
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            # Add to ChromaDB
            pdf_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.warning(f"PDF indexed with document_id: {document_id}")
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return PDFResponse(
                success=True,
                document_id=document_id,
                page_count=page_count,
                message=f"PDF processed successfully with {len(chunks)} text chunks"
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            logger.error(traceback.format_exc())
            # Clean up temp file
            os.unlink(temp_file.name)
            return PDFResponse(success=False, message=f"Error processing PDF: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in PDF upload: {str(e)}")
        logger.error(traceback.format_exc())
        return PDFResponse(success=False, message=f"Error: {str(e)}")

@app.post("/documents")
async def add_document(document: Dict[str, Any]):
    """
    Add a document to the document store for RAG
    """
    try:
        # Add to Haystack document store
        doc = Document(
            content=document.get("content", ""),
            meta=document.get("meta", {})
        )
        document_store.write_documents([doc])
        
        # Also add to ChromaDB for consistency
        document_id = str(uuid.uuid4())
        pdf_collection.add(
            documents=[document.get("content", "")],
            metadatas=[document.get("meta", {})],
            ids=[document_id]
        )
        
        return {"status": "success", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.warning("Starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 