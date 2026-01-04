"""
Configuration settings for the Admission Assistant Chatbot.

This module centralizes all configuration parameters to ensure
consistent behavior across the application.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# LLM Model for text generation - using phi3 (can be changed to "tinyllama" if preferred)
LLM_MODEL = os.getenv("LLM_MODEL", "phi3")  # Phi-3 or tinyllama model name in Ollama
# Embedding model - nomic-embed-text is recommended, but phi3/tinyllama can be used as fallback
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")  # Embedding model name in Ollama

# LLM Generation Parameters
TEMPERATURE = 0.2  # Low temperature to reduce hallucinations
MAX_TOKENS = 512  # Reasonable response length
TOP_P = 0.9  # Nucleus sampling parameter

# RAG Configuration
CHUNK_SIZE = 800  # Characters per chunk (increased to preserve more context)
CHUNK_OVERLAP = 100  # Overlap between chunks to preserve context
TOP_K = 5  # Number of relevant chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.1  # Minimum similarity score for retrieval (very low to ensure we get results)

# Vector Store Configuration
FAISS_INDEX_NAME = "admission_index"
# Embedding dimension: 768 for nomic-embed-text, 4096 for phi3 embeddings
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # Default for nomic-embed-text

# Prompt Template
SYSTEM_PROMPT = """You are an Admission Assistant for K J Somaiya Institute of Technology. 
Your role is to provide accurate, official admission information based ONLY on the provided context.

STRICT RULES:
1. Answer ONLY from the provided context below - extract exact information
2. If the context contains cutoff marks, percentages (like "Max 99.46%, Min 97.96%"), fees, or any numerical data, you MUST provide those exact numbers
3. Do NOT say "I do not have this information" if the context contains relevant data - extract and present it
4. When asked about cutoffs, carefully search the context for percentages, "Max", "Min", "CAP", "Minority" - these indicate cutoff marks
5. Be formal, clear, and professional
6. If you see data like "Max 99.46%, Min 97.96%" in the context, that IS cutoff information - present it clearly

CRITICAL ADMISSION TYPE RULE:
- If the user's question mentions "DSY", "Direct Second Year", "second year", "DSE", or "lateral entry", ONLY use information from the "DIRECT SECOND YEAR (DSY)" section of the context
- If the user's question does NOT mention DSY/Direct Second Year, ONLY use information from the "FIRST YEAR (FY)" section of the context
- Do NOT mix FY and DSY information unless the user explicitly asks about both
- When user asks general questions (like "cutoff", "fees", "eligibility") without specifying DSY, provide FY information only

Context from official admission data:
{context}

User Question: {question}

IMPORTANT: 
1. Read the context carefully and identify if the question is about FY or DSY
2. If percentages, Max/Min values, or branch names with numbers appear in the relevant section, extract and present them clearly
3. Do NOT say the information is not available if it appears in the context above
4. Default to FY information unless DSY is explicitly mentioned

Provide a clear, accurate answer based ONLY on the relevant section (FY or DSY) of the context above:"""

