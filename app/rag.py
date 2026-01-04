"""
RAG (Retrieval-Augmented Generation) Pipeline

This module implements the complete RAG pipeline:
1. Retrieve relevant chunks from vector store
2. Generate answer using LLM with retrieved context

This ensures the chatbot only uses information from the admission data,
preventing hallucinations.
"""

from typing import List, Optional
import requests
from app.config import (
    OLLAMA_BASE_URL, LLM_MODEL, TEMPERATURE, MAX_TOKENS, TOP_P,
    SYSTEM_PROMPT, TOP_K, SIMILARITY_THRESHOLD
)
from app.embeddings import OllamaEmbeddings
from app.vector_store import FAISSVectorStore


class RAGPipeline:
    """
    RAG Pipeline for admission assistant chatbot.
    
    WHY RAG: Retrieval-Augmented Generation combines the best of both worlds:
    - Retrieval finds relevant information from the knowledge base
    - Generation creates natural language responses
    - Together, they ensure grounded, accurate answers without hallucinations
    """
    
    def __init__(self, vector_store: FAISSVectorStore, embeddings: OllamaEmbeddings):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: FAISS vector store instance
            embeddings: Ollama embeddings client
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=top_k)
        
        # Extract chunks (without similarity scores)
        chunks = [chunk for chunk, score in results]
        
        return chunks
    
    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer
        """
        # Combine chunks into context
        context = "\n\n".join(context_chunks)
        
        # Format prompt
        prompt = SYSTEM_PROMPT.format(context=context, question=query)
        
        # Prepare LLM request
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
                "top_p": TOP_P
            }
        }
        
        try:
            response = requests.post(self.llm_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Please ensure Ollama is running and the model is installed:\n"
                f"  ollama pull {LLM_MODEL}"
            )
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"Error generating response: {str(e)}")
    
    def ask(self, query: str) -> str:
        """
        Complete RAG pipeline: retrieve + generate.
        
        This is the main method used by the API endpoint.
        
        Args:
            query: User question
            
        Returns:
            Generated answer
        """
        # Check if vector store is initialized
        if self.vector_store.is_empty():
            return (
                "The admission database is not initialized. "
                "Please load admission data first."
            )
        
        # Retrieve relevant chunks - use very permissive search
        query_embedding = self.embeddings.embed_query(query)
        
        # First try with threshold
        results = self.vector_store.search(query_embedding, k=TOP_K, threshold=SIMILARITY_THRESHOLD)
        chunks = [chunk for chunk, score in results]
        
        # If no chunks or very few, get top-k regardless of threshold
        if len(chunks) < 2:
            results = self.vector_store.search(query_embedding, k=TOP_K, threshold=0.0)
            chunks = [chunk for chunk, score in results[:TOP_K]]
        
        # Filter chunks based on admission type (FY vs DSY)
        query_lower = query.lower()
        is_dsy_query = any(keyword in query_lower for keyword in ['dsy', 'direct second year', 'second year', 'dse', 'lateral entry', 'diploma'])
        
        if is_dsy_query:
            # Filter to only DSY chunks
            chunks = [chunk for chunk in chunks 
                     if any(kw in chunk.lower() for kw in ['direct second year', 'dsy', 'dse', 'diploma', 'second year admission'])]
        else:
            # Filter to only FY chunks (default)
            chunks = [chunk for chunk in chunks 
                     if not any(kw in chunk.lower() for kw in ['direct second year', 'dsy', 'dse']) 
                     or any(kw in chunk.lower() for kw in ['first year', 'fy', 'hsc', 'mht-cet', 'jee main'])]
        
        # Keyword-based fallback for cutoff queries
        if any(keyword in query_lower for keyword in ['cutoff', 'cut-off', 'cut off', 'percentile', 'percentage', 'marks']):
            # Get all chunks from vector store
            try:
                all_chunks = self.vector_store.chunks if hasattr(self.vector_store, 'chunks') else []
                if all_chunks:
                    # Find chunks containing cutoff-related keywords, filtered by admission type
                    cutoff_chunks = [chunk for chunk in all_chunks 
                                    if any(kw in chunk.lower() for kw in ['cutoff', 'cut-off', 'cut off', 'percentile', 'percentage', 'max', 'min', 'mht cet'])]
                    
                    # Apply admission type filter
                    if is_dsy_query:
                        cutoff_chunks = [chunk for chunk in cutoff_chunks 
                                        if any(kw in chunk.lower() for kw in ['direct second year', 'dsy', 'dse'])]
                    else:
                        cutoff_chunks = [chunk for chunk in cutoff_chunks 
                                        if not any(kw in chunk.lower() for kw in ['direct second year', 'dsy', 'dse'])
                                        or any(kw in chunk.lower() for kw in ['first year', 'fy'])]
                    
                    if cutoff_chunks:
                        # Add cutoff chunks to the results, avoiding duplicates
                        existing_texts = set(chunks)
                        for chunk in cutoff_chunks[:5]:  # Get up to 5 cutoff chunks
                            if chunk not in existing_texts:
                                chunks.append(chunk)
                                existing_texts.add(chunk)
            except:
                pass  # If we can't access chunks, continue with vector search results
        
        # If still no chunks, return default response
        if not chunks:
            return (
                "I do not have this information in the official admission data. "
                "Please refer to the official college website or contact the admission office."
            )
        
        # Generate answer
        answer = self.generate(query, chunks)
        
        return answer
    
    def check_models_available(self) -> dict:
        """
        Check if required Ollama models are available.
        
        Returns:
            Dictionary with availability status for each model
        """
        try:
            response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            return {
                "llm_model": any(LLM_MODEL in name for name in model_names),
                "embedding_model": any(EMBEDDING_MODEL in name for name in model_names),
                "ollama_available": True
            }
        except:
            return {
                "llm_model": False,
                "embedding_model": False,
                "ollama_available": False
            }


