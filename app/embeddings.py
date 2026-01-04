"""
Embeddings Module

This module handles generating embeddings using Ollama's nomic-embed-text model.
Embeddings convert text into numerical vectors that capture semantic meaning,
enabling similarity search in the vector database.
"""

import requests
from typing import List, Optional
from app.config import OLLAMA_BASE_URL, EMBEDDING_MODEL


class OllamaEmbeddings:
    """
    Client for generating embeddings using Ollama's embedding models.
    
    WHY: Embeddings convert text into high-dimensional vectors that capture
    semantic meaning. Similar texts have similar vectors, enabling efficient
    similarity search for RAG retrieval.
    """
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = EMBEDDING_MODEL):
        """
        Initialize Ollama embeddings client.
        
        Args:
            base_url: Base URL for Ollama API
            model: Name of the embedding model in Ollama (can be nomic-embed-text, phi3, or tinyllama)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embedding_url = f"{self.base_url}/api/embeddings"
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single text query.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            requests.RequestException: If Ollama API call fails
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.embedding_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running and the model is installed:\n"
                f"  ollama pull {self.model}"
            )
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"Error generating embedding: {str(e)}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Processes in batches to avoid overwhelming the API.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.embed_query(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def check_model_available(self) -> bool:
        """
        Check if the embedding model is available in Ollama.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            return any(self.model in name for name in model_names)
        except:
            return False

