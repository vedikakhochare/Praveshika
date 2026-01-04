"""
Vector Store Module

This module manages the FAISS vector database for storing and retrieving
document embeddings. FAISS enables fast similarity search for RAG retrieval.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from app.config import VECTOR_STORE_DIR, FAISS_INDEX_NAME, EMBEDDING_DIMENSION, TOP_K, SIMILARITY_THRESHOLD


class FAISSVectorStore:
    """
    FAISS-based vector store for document embeddings.
    
    WHY: FAISS provides efficient similarity search on high-dimensional vectors.
    It enables fast retrieval of relevant document chunks for RAG, which is
    critical for real-time chatbot responses.
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []  # Store original text chunks (make it accessible)
        self.index_path = VECTOR_STORE_DIR / f"{FAISS_INDEX_NAME}.index"
        self.chunks_path = VECTOR_STORE_DIR / f"{FAISS_INDEX_NAME}_chunks.pkl"
    
    def create_index(self):
        """
        Create a new FAISS index.
        
        Uses L2 (Euclidean) distance for similarity search.
        """
        # L2 index is suitable for normalized embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
    
    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str]):
        """
        Add embeddings and corresponding text chunks to the index.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of corresponding text chunks
        """
        if not embeddings or not chunks:
            raise ValueError("Embeddings and chunks cannot be empty")
        
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Create index if it doesn't exist
        if self.index is None:
            self.create_index()
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity (L2 normalization)
        # This allows using L2 distance for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store chunks
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: List[float], k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity score (0-1, higher is more similar)
            
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Convert distances to similarity scores (1 - normalized distance)
        # Since we're using normalized vectors, L2 distance approximates cosine distance
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                # Convert distance to similarity (higher is better)
                # For normalized vectors: similarity ≈ 1 - (distance / 2)
                similarity = max(0.0, 1.0 - (dist / 2.0))
                
                # Always include top-k results, but filter by threshold if provided
                if similarity >= threshold:
                    results.append((self.chunks[idx], similarity))
                elif len(results) < k and threshold <= 0.2:  # If threshold is very low, include top-k anyway
                    results.append((self.chunks[idx], similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def save(self):
        """
        Save the index and chunks to disk.
        
        WHY: Persistence allows reusing the vector store without
        regenerating embeddings, which is time-consuming.
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load(self) -> bool:
        """
        Load the index and chunks from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load chunks
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def is_empty(self) -> bool:
        """
        Check if the vector store is empty.
        
        Returns:
            True if empty, False otherwise
        """
        return self.index is None or self.index.ntotal == 0
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats (count, dimension, etc.)
        """
        if self.index is None:
            return {"count": 0, "dimension": self.dimension}
        
        return {
            "count": self.index.ntotal,
            "dimension": self.dimension,
            "chunks_count": len(self.chunks)
        }

