"""
Text Chunking Module

This module implements intelligent text chunking for RAG.
Chunks are created with overlap to preserve context across boundaries,
ensuring that information isn't lost at split points.
"""

from typing import List
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for RAG processing.
    
    WHY: Large documents need to be split into smaller pieces that fit
    into the LLM's context window. Overlap ensures context continuity
    at chunk boundaries, preventing information loss.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    # Clean and normalize text
    text = text.strip()
    
    # If text is smaller than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is the last chunk, take remaining text
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to break at sentence boundary (prefer . ! ? followed by space)
        # This prevents splitting mid-sentence
        break_point = end
        
        # Look backwards for sentence boundary
        for i in range(end, max(start, end - 200), -1):
            if i < len(text) - 1:
                if text[i] in '.!?' and text[i+1] == ' ':
                    break_point = i + 1
                    break
        
        # If no sentence boundary found, try paragraph boundary
        if break_point == end:
            for i in range(end, max(start, end - 100), -1):
                if i < len(text) - 1 and text[i] == '\n' and text[i+1] == '\n':
                    break_point = i + 1
                    break
        
        # Extract chunk
        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = break_point - chunk_overlap
        if start < 0:
            start = 0
    
    return chunks


def chunk_text_simple(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking (fallback method).
    
    Used when sentence/paragraph boundaries are not found.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    text = text.strip()
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks





