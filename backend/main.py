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
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
import PyPDF2
import google.generativeai as genai
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from chunkr_ai import Chunkr

from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
import sys
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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

# Initialize ChromaDB client for chat history and documents
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chat_collection = chroma_client.get_or_create_collection("chat_history")
    pdf_collection = chroma_client.get_or_create_collection("pdf_documents")
    
    # Initialize Haystack ChromaDocumentStore
    document_store = ChromaDocumentStore(
        persist_path="./chroma_db"
    )
    logger.warning("ChromaDB and Haystack ChromaDocumentStore initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# Models available for selection
AVAILABLE_MODELS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
    "gemini-pro": "gemini-2.0-pro-exp"  # Updated Gemini Pro model name
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
                # Use Google Gemini for generation through Haystack
                logger.warning("Using Google Gemini model through Haystack")
                
                # Create a chat pipeline
                chat_pipeline = Pipeline()
                
                # Add Google Gemini generator
                gemini_generator = GoogleAIGeminiGenerator(
                    api_key=Secret.from_token(gemini_api_key),
                    model=AVAILABLE_MODELS[request.model]
                )
                chat_pipeline.add_component("generator", gemini_generator)
                
                # Extract the last user message
                last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
                if not last_user_message:
                    raise ValueError("No user message found")
                
                # Run the generator
                result = chat_pipeline.run(
                    data={
                        "generator": {"prompt": last_user_message}
                    }
                )
                
                # Extract the response
                response_text = result["generator"]["replies"][0]
                logger.warning("Gemini model completed successfully through Haystack")
            except Exception as e:
                logger.error(f"Error using Gemini model through Haystack: {str(e)}")
                # Fall back to direct API call if Haystack integration fails
                try:
                    logger.warning("Falling back to direct Gemini API call")
                    gemini_model = genai.GenerativeModel(model_name=AVAILABLE_MODELS[request.model])
                    
                    # Convert to Gemini's format
                    gemini_messages = []
                    for msg in request.messages:
                        role = "user" if msg.role == "user" else "model"
                        gemini_messages.append({"role": role, "parts": [msg.content]})
                    
                    # Generate response
                    chat = gemini_model.start_chat(history=gemini_messages[:-1])
                    response = chat.send_message(gemini_messages[-1]["parts"][0])
                    response_text = response.text
                    logger.warning("Gemini model completed successfully with direct API call")
                except Exception as fallback_error:
                    logger.error(f"Error in Gemini fallback: {str(fallback_error)}")
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
        
        # Check which model to use for generation
        if request.model == "gemini-pro":
            if not gemini_api_key:
                raise ValueError("Google Gemini API key not found in environment variables")
            
            try:
                # Use Google Gemini for RAG generation
                logger.warning("Using Google Gemini model for RAG")
                
                # Create a RAG pipeline with Haystack and ChromaDB
                rag_pipeline = Pipeline()
                
                # Add retriever component from ChromaDocumentStore
                retriever = ChromaQueryTextRetriever(
                    document_store=document_store,
                    top_k=5  # Set a reasonable default
                )
                rag_pipeline.add_component("retriever", retriever)

                # Add prompt builder
                rag_pipeline.add_component("prompt_builder", PromptBuilder(
                    template="""
                    Answer the following question based on the provided context. 
                    If the answer is not in the context, say that you don't know.
                    
                    Context:
                    {% for document in documents %}
                    {{ document.content }}
                    {% endfor %}
                    
                    Question: {{query}}
                    
                    Answer:
                    """
                ))
                
                # Add Google Gemini generator
                gemini_generator = GoogleAIGeminiGenerator(
                    api_key=Secret.from_token(gemini_api_key),
                    model=AVAILABLE_MODELS[request.model]
                )
                rag_pipeline.add_component("generator", gemini_generator)
                
                # Connect components
                rag_pipeline.connect("retriever", "prompt_builder.documents")
                rag_pipeline.connect("prompt_builder", "generator")
                
                # Run the pipeline
                try:
                    result = rag_pipeline.run(
                        data={
                            "retriever": {"query": user_query},
                            "prompt_builder": {"query": user_query}
                        }
                    )
                    
                    # Extract the response
                    response_text = result["generator"]["replies"][0]
                    logger.warning("Gemini RAG pipeline completed successfully")
                except Exception as e:
                    logger.error(f"Error running Haystack pipeline: {str(e)}")
                    # If no documents were found, fall back to regular chat
                    if "No documents found" in str(e):
                        logger.warning("No documents found for RAG, falling back to regular chat")
                        return await chat(request)
                    raise
                
                return {
                    "response": response_text,
                    "model": request.model
                }
            except Exception as e:
                logger.error(f"Error using Gemini model for RAG: {str(e)}")
                # If no documents were found, fall back to regular chat
                if "No documents found" in str(e):
                    logger.warning("No documents found for RAG, falling back to regular chat")
                    return await chat(request)
                raise
        else:
            # Check if we have a valid OpenAI API key
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            # Create a complete Haystack RAG pipeline with ChromaDB
            try:
                logger.warning(f"Using OpenAI model {AVAILABLE_MODELS[request.model]} for RAG")
                
                # Create the RAG pipeline
                rag_pipeline = Pipeline()
                
                # Add retriever component from ChromaDocumentStore
                retriever = ChromaQueryTextRetriever(
                    document_store=document_store,
                    
                )
                rag_pipeline.add_component("retriever", retriever)
                
                # Add prompt builder
                rag_pipeline.add_component("prompt_builder", PromptBuilder(
                    template="""
                    Answer the following question based on the provided context. 
                    If the answer is not in the context, say that you don't know.
                    
                    Context:
                    {% for document in documents %}
                    {{ document.content }}
                    {% endfor %}
                    
                    Question: {{query}}
                    
                    Answer:
                    """
                ))
                
                # Add generator
                rag_pipeline.add_component("generator", OpenAIGenerator(
                    api_key=api_key_secret,
                    model=AVAILABLE_MODELS[request.model]
                ))
                
                # Connect components
                rag_pipeline.connect("retriever", "prompt_builder.documents")
                rag_pipeline.connect("prompt_builder", "generator")
                
                # Run the pipeline
                try:
                    result = rag_pipeline.run(
                        data={
                            "retriever": {"query": user_query, "top_k": 5, "include_metadata": True, },
                            "prompt_builder": {"query": user_query}
                        }
                    )
                    
                    logger.warning(f"Result: {result}")
                    
                    # Extract the response
                    response_text = result["generator"]["replies"][0]
                    logger.warning("OpenAI RAG pipeline completed successfully")
                except Exception as e:
                    logger.error(f"Error running Haystack pipeline: {str(e)}")
                    # If no documents were found, fall back to regular chat
                    if "No documents found" in str(e):
                        logger.warning("No documents found for RAG, falling back to regular chat")
                        return await chat(request)
                    raise
                
                return {
                    "response": response_text,
                    "model": request.model
                }
            except Exception as e:
                logger.error(f"Error in Haystack RAG pipeline: {str(e)}")
                # If no documents were found, fall back to regular chat
                if "No documents found" in str(e):
                    logger.warning("No documents found for RAG, falling back to regular chat")
                    return await chat(request)
                raise
    
    except Exception as e:
        logger.error(f"Error in RAG endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf", response_model=PDFResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file for RAG using Chunkr
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
        
        try:
            # Initialize Chunkr client
            chunkr_api_key = os.environ.get("CHUNKR_API_KEY", "")
            if not chunkr_api_key:
                logger.warning("CHUNKR_API_KEY not found in environment variables. Make sure it's set or pass it directly.")
            
            chunkr = Chunkr(api_key=chunkr_api_key if chunkr_api_key else None)
            
            # Configure chunking parameters
            
            # Process PDF with Chunkr
            task = await chunkr.upload(temp_file.name)

            # Wait for the task to complete and get the chunks
            chunks = task.output.chunks
            logger.warning(f"Created {len(chunks)} text chunks from PDF using Chunkr")
            
            # Add chunks to ChromaDB
            document_id = str(uuid.uuid4())
            
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            
            # Debug each chunk in detail
            for i, chunk in enumerate(chunks):
                logger.warning(f"Processing chunk {i+1}/{len(chunks)}")
                logger.warning(f"Chunk ID: {chunk.chunk_id}")
                logger.warning(f"Chunk length: {chunk.chunk_length}")
                logger.warning(f"Number of segments: {len(chunk.segments)}")
                
                # The 'embed' property contains the markdown text of all segments
                chunk_text = chunk.embed
                logger.warning(f"Chunk text: {chunk_text[:100]}..." if len(chunk_text) > 100 else chunk_text)
                
                # Get page numbers from segments
                page_numbers = set(segment.page_number for segment in chunk.segments if hasattr(segment, 'page_number'))
                pages_str = ",".join(str(page) for page in page_numbers)
                logger.warning(f"Pages in chunk: {pages_str}")
                
                # Extract metadata from chunk
                chunk_metadata = {
                    "source": file.filename,
                    "chunk_id": chunk.chunk_id,
                    "pages": pages_str,
                    "chunk_index": i + 1
                }
                
                texts.append(chunk_text)
                metadatas.append(chunk_metadata)
            
            ids = [f"{document_id}_{i}" for i in range(len(texts))]
            
            # Add to ChromaDB
            pdf_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.warning(f"PDF indexed with document_id: {document_id}")
            
            # Add to Haystack ChromaDocumentStore
            haystack_docs = []
            for i, chunk in enumerate(chunks):
                haystack_docs.append(
                    Document(
                        content=chunk.embed,
                        meta=metadatas[i]
                    )
                )
            
            # Create a document writer and write documents to the store
            writer = DocumentWriter(document_store=document_store)
            writer.run(documents=haystack_docs)
            logger.warning(f"Added {len(haystack_docs)} documents to Haystack ChromaDocumentStore")
            
            return PDFResponse(
                success=True,
                document_id=document_id,
                page_count=len(set(segment.page_number for chunk in chunks for segment in chunk.segments if hasattr(segment, 'page_number'))),
                message=f"PDF processed successfully with {len(chunks)} text chunks using Chunkr"
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF with Chunkr: {str(e)}")
            logger.error(traceback.format_exc())
            return PDFResponse(success=False, message=f"Error processing PDF: {str(e)}")
        finally:
            # Clean up resources
            try:
                await chunkr.close()
                os.unlink(temp_file.name)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
            
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
        # Add to ChromaDB
        document_id = str(uuid.uuid4())
        pdf_collection.add(
            documents=[document.get("content", "")],
            metadatas=[document.get("meta", {})],
            ids=[document_id]
        )
        
        # Add to Haystack ChromaDocumentStore
        doc = Document(
            content=document.get("content", ""),
            meta=document.get("meta", {})
        )
        
        # Create a document writer and write documents to the store
        writer = DocumentWriter(document_store=document_store)
        writer.run(documents=[doc])
        
        logger.warning(f"Document added to ChromaDB and Haystack with ID: {document_id}")
        return {"status": "success", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.warning("Starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, loop="asyncio", log_level="debug") 