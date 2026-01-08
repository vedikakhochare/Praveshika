"""
FastAPI Main Application

This module defines the FastAPI application with endpoints for:
- Chatbot queries (/ask)
- Health checks (/health)
- Data loading (/load-data)
- Status checks (/status)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import json
from pathlib import Path

from app.config import DATA_DIR, TOP_K, TOP_K, SIMILARITY_THRESHOLD
from app.data_ingestion import load_admission_data, get_available_data_files
from app.text_chunking import chunk_text
from app.embeddings import OllamaEmbeddings
from app.vector_store import FAISSVectorStore
from app.rag import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Admission Assistant Chatbot",
    description="Local-only RAG-based chatbot for engineering college admissions",
    version="1.0.0"
)

# Global instances (initialized on startup)
vector_store: Optional[FAISSVectorStore] = None
embeddings: Optional[OllamaEmbeddings] = None
rag_pipeline: Optional[RAGPipeline] = None


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class LoadDataRequest(BaseModel):
    filename: Optional[str] = None


class StatusResponse(BaseModel):
    vector_store_loaded: bool
    vector_store_stats: dict
    models_available: dict
    available_data_files: list


@app.on_event("startup")
async def startup_event():
    """
    Initialize components on application startup.
    
    WHY: Pre-initialize vector store and RAG pipeline to avoid
    repeated initialization on each request.
    """
    global vector_store, embeddings, rag_pipeline
    
    # Initialize components
    embeddings = OllamaEmbeddings()
    vector_store = FAISSVectorStore()
    
    # Try to load existing vector store
    if vector_store.load():
        print(f"✓ Loaded existing vector store with {vector_store.index.ntotal} chunks")
    else:
        print("ℹ No existing vector store found. Load admission data to initialize.")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(vector_store, embeddings)
    
    print("✓ Application started successfully")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the frontend HTML page.
    """
    html_path = Path(__file__).parent.parent / "templates" / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>Admission Assistant</title></head>
            <body>
                <h1>Admission Assistant Chatbot</h1>
                <p>Frontend not found. Please ensure templates/index.html exists.</p>
                <p>API is available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Main endpoint for asking questions to the chatbot with streaming support.
    
    This endpoint:
    1. Takes a user question
    2. Uses RAG to retrieve relevant context
    3. Streams answer using LLaMA 3.2
    4. Returns streaming response
    
    Args:
        request: QueryRequest with user question
        
    Returns:
        StreamingResponse with generated answer chunks
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    def generate():
        try:
            # Check if vector store is initialized
            if rag_pipeline.vector_store.is_empty():
                yield f"data: {json.dumps({'chunk': 'The admission database is not initialized. Please load admission data first.', 'done': True})}\n\n"
                return
            
            # Retrieve relevant chunks using RAG pipeline's retrieve method
            chunks = rag_pipeline.retrieve(request.question.strip(), top_k=TOP_K)
            
            if not chunks:
                yield f"data: {json.dumps({'chunk': 'I do not have this information in the official admission data. Please refer to the official college website or contact the admission office.', 'done': True})}\n\n"
                return
            
            # Stream the answer
            for text_chunk in rag_pipeline.generate_stream(request.question.strip(), chunks):
                if text_chunk:
                    yield f"data: {json.dumps({'chunk': text_chunk, 'done': False})}\n\n"
            
            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
            
        except ConnectionError as e:
            yield f"data: {json.dumps({'chunk': f'Error: {str(e)}', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'chunk': f'Error processing query: {str(e)}', 'done': True})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/load-data")
async def load_data(request: LoadDataRequest):
    """
    Load and process admission data file.
    
    This endpoint:
    1. Reads the admission document (DOCX or TXT)
    2. Chunks the text
    3. Generates embeddings
    4. Stores in FAISS vector database
    
    Args:
        request: LoadDataRequest with optional filename
        
    Returns:
        Success message with statistics
    """
    global vector_store, embeddings, rag_pipeline
    
    if embeddings is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    try:
        # Load text from file
        text = load_admission_data(request.filename)
        
        # Chunk text
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from file")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = embeddings.embed_documents(chunks)
        
        # Create new vector store
        vector_store.create_index()
        vector_store.add_embeddings(chunk_embeddings, chunks)
        vector_store.save()
        
        # Reinitialize RAG pipeline
        rag_pipeline = RAGPipeline(vector_store, embeddings)
        
        stats = vector_store.get_stats()
        
        return {
            "message": "Data loaded successfully",
            "chunks_processed": stats["count"],
            "filename": request.filename or "auto-detected"
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get system status and statistics.
    
    Returns:
        StatusResponse with vector store status, model availability, etc.
    """
    if vector_store is None or rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    stats = vector_store.get_stats() if not vector_store.is_empty() else {}
    models_available = rag_pipeline.check_models_available()
    available_files = get_available_data_files()
    
    return StatusResponse(
        vector_store_loaded=not vector_store.is_empty(),
        vector_store_stats=stats,
        models_available=models_available,
        available_data_files=available_files
    )


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy", "service": "admission-assistant-chatbot"}

