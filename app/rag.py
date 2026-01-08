"""
RAG (Retrieval-Augmented Generation) Pipeline

This module implements the complete RAG pipeline:
1. Retrieve relevant chunks from vector store
2. Generate answer using LLM with retrieved context

This ensures the chatbot only uses information from the admission data,
preventing hallucinations.
"""

from typing import List, Optional, Generator, Tuple
import requests
import json
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
    
    def _detect_branch_and_admission_type(self, query: str) -> Tuple[Optional[str], bool]:
        """
        Detect branch name and admission type (FY/DSY) from query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (branch_name or None, is_dsy: bool)
        """
        query_lower = query.lower()
        
        # Detect admission type
        is_dsy = any(keyword in query_lower for keyword in [
            'dsy', 'direct second year', 'second year', 'dse', 
            'lateral entry', 'diploma', 'direct se'
        ])
        
        # Branch name mappings (query patterns -> full branch names)
        # Order matters - more specific patterns first
        branch_patterns = [
            ('artificial intelligence and data science', ['ai&ds', 'ai and ds', 'artificial intelligence and data science', 'aids', 'ai/ds']),
            ('computer engineering', ['computer engineering', 'computer eng', 'comp eng']),
            ('information technology', ['information technology', 'it branch', 'it engineering']),
            ('electronics and telecommunication', ['electronics and telecommunication', 'extc', 'e&tc', 'electronics and telecom'])
        ]
        
        # Also check for shorter patterns but be more careful
        short_patterns = {
            'artificial intelligence and data science': ['ai&ds', 'ai and ds'],
            'computer engineering': ['computer', 'cse'],
            'information technology': ['it'],
            'electronics and telecommunication': ['electronics', 'telecommunication', 'etc']
        }
        
        detected_branch = None
        
        # First check for specific patterns
        for branch_name, patterns in branch_patterns:
            if any(pattern in query_lower for pattern in patterns):
                detected_branch = branch_name
                break
        
        # If no specific match, check short patterns (but be VERY strict - only if clearly about a branch)
        if not detected_branch:
            # Only detect branch from short patterns if:
            # 1. The pattern appears with explicit branch context (e.g., "computer cutoff", "IT fees")
            # 2. The query is very short AND contains branch-specific keywords
            query_words = query_lower.split()
            for branch_name, patterns in short_patterns.items():
                for pattern in patterns:
                    if pattern in query_lower:
                        # Check if query explicitly mentions branch context
                        explicit_branch_context = any(phrase in query_lower for phrase in [
                            f'{pattern} cutoff', f'{pattern} cut-off', f'{pattern} cut off',
                            f'{pattern} fees', f'{pattern} eligibility', f'{pattern} admission',
                            f'{pattern} branch', f'{pattern} course', f'for {pattern}',
                            f'in {pattern}', f'{pattern} for', f'{pattern} in'
                        ])
                        
                        # Only detect if there's explicit branch context
                        if explicit_branch_context:
                            detected_branch = branch_name
                            break
                if detected_branch:
                    break
        
        return detected_branch, is_dsy
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        """
        Retrieve relevant chunks for a query with branch and admission type filtering.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks filtered by branch and admission type
        """
        # Detect branch and admission type
        branch_name, is_dsy = self._detect_branch_and_admission_type(query)
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search vector store (get more results for filtering)
        results = self.vector_store.search(query_embedding, k=top_k * 3, threshold=0.0)
        
        # Filter chunks based on admission type (FY vs DSY)
        filtered_chunks = []
        for chunk, score in results:
            chunk_lower = chunk.lower()
            
            # Admission type filtering
            if is_dsy:
                # Only include DSY chunks
                if not any(kw in chunk_lower for kw in ['direct second year', 'dsy', 'dse', 'diploma', 'second year admission']):
                    continue
            else:
                # Only include FY chunks (exclude DSY chunks)
                if any(kw in chunk_lower for kw in ['direct second year', 'dsy', 'dse']) and \
                   not any(kw in chunk_lower for kw in ['first year', 'fy', 'hsc', 'mht-cet', 'jee main']):
                    continue
            
            # Branch filtering (ONLY if branch is explicitly specified)
            if branch_name:
                # Check if chunk contains the branch name or related keywords
                branch_keywords = {
                    'artificial intelligence and data science': [
                        'artificial intelligence and data science', 'ai&ds', 'ai and ds', 
                        'aids', 'branch: artificial intelligence'
                    ],
                    'computer engineering': [
                        'computer engineering', 'branch: computer engineering', 
                        'computer eng'
                    ],
                    'information technology': [
                        'information technology', 'branch: information technology', 
                        'it branch'
                    ],
                    'electronics and telecommunication': [
                        'electronics and telecommunication', 'extc', 
                        'branch: electronics'
                    ]
                }
                
                keywords = branch_keywords.get(branch_name, [])
                chunk_mentions_branch = any(kw in chunk_lower for kw in keywords)
                
                # If branch is specified, ONLY include chunks that mention that branch
                # This is strict filtering - if user asked about a specific branch, 
                # we should only show information about that branch
                if not chunk_mentions_branch:
                    continue  # Skip chunks that don't mention the specified branch
            # If no branch is specified, include all chunks (let LLM provide general info)
            
            filtered_chunks.append((chunk, score))
        
        # If no filtered chunks, fallback to original results but still apply admission type filter
        if not filtered_chunks:
            for chunk, score in results:
                chunk_lower = chunk.lower()
                if is_dsy:
                    if any(kw in chunk_lower for kw in ['direct second year', 'dsy', 'dse', 'diploma', 'second year admission']):
                        filtered_chunks.append((chunk, score))
                else:
                    if not any(kw in chunk_lower for kw in ['direct second year', 'dsy', 'dse']) or \
                       any(kw in chunk_lower for kw in ['first year', 'fy', 'hsc', 'mht-cet', 'jee main']):
                        filtered_chunks.append((chunk, score))
        
        # Sort by similarity and return top_k
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)
        chunks = [chunk for chunk, score in filtered_chunks[:top_k]]
        
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
    
    def generate_stream(self, query: str, context_chunks: List[str]):
        """
        Generate answer using LLM with streaming support.
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            
        Yields:
            Text chunks as they are generated
        """
        # Combine chunks into context
        context = "\n\n".join(context_chunks)
        
        # Format prompt
        prompt = SYSTEM_PROMPT.format(context=context, question=query)
        
        # Prepare LLM request with streaming
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
                "top_p": TOP_P
            }
        }
        
        try:
            response = requests.post(self.llm_url, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_data = json.loads(line)
                        if "response" in json_data:
                            yield json_data["response"]
                        if json_data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
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


