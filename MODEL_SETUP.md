# Model Configuration Guide

## Current Configuration

The system is now configured to use **Phi-3** (or TinyLlama) instead of LLaMA 3.2.

## Available Models

You have the following models installed:
- ✅ **phi3:latest** (2.2 GB) - Recommended for better quality
- ✅ **tinyllama:latest** (637 MB) - Smaller, faster, but less capable

## Model Selection

### Using Phi-3 (Default - Recommended)
The system is already configured to use `phi3` by default. No changes needed!

### Using TinyLlama Instead
If you want to use TinyLlama instead, set the environment variable:

**Windows PowerShell:**
```powershell
$env:LLM_MODEL="tinyllama"
python run.py
```

**Windows CMD:**
```cmd
set LLM_MODEL=tinyllama
python run.py
```

**Linux/Mac:**
```bash
export LLM_MODEL=tinyllama
python run.py
```

## Embedding Models

### Recommended: nomic-embed-text
For best results, install the dedicated embedding model:
```bash
ollama pull nomic-embed-text
```

### Fallback: Using LLM for Embeddings
If you don't have `nomic-embed-text`, the system can use your LLM model (phi3 or tinyllama) for embeddings, but:
- ⚠️ Results may be less accurate
- ⚠️ Embeddings will be larger (4096 dimensions vs 768)
- ⚠️ Slower processing

To use phi3 for embeddings:
```bash
# Windows PowerShell
$env:EMBEDDING_MODEL="phi3"
$env:EMBEDDING_DIMENSION="4096"

# Linux/Mac
export EMBEDDING_MODEL="phi3"
export EMBEDDING_DIMENSION="4096"
```

## Quick Start

1. **Start with Phi-3 (already configured):**
   ```bash
   python run.py
   ```

2. **If you want better embeddings, install nomic-embed-text:**
   ```bash
   ollama pull nomic-embed-text
   ```
   The system will automatically use it if available.

3. **To switch to TinyLlama:**
   ```bash
   # Windows PowerShell
   $env:LLM_MODEL="tinyllama"
   python run.py
   ```

## Performance Comparison

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| **phi3** | 2.2 GB | ⭐⭐⭐⭐ | Medium | **Recommended** - Best balance |
| **tinyllama** | 637 MB | ⭐⭐ | Fast | Good for testing/limited resources |
| **nomic-embed-text** | ~274 MB | ⭐⭐⭐⭐⭐ | Fast | **Best for embeddings** |

## Troubleshooting

### Model Not Found Error
If you get "model not found" errors:
1. Check installed models: `ollama list`
2. Pull the model: `ollama pull phi3` or `ollama pull tinyllama`
3. Restart the server

### Slow Performance
- Use `tinyllama` for faster responses (but lower quality)
- Install `nomic-embed-text` for faster embeddings
- Reduce `CHUNK_SIZE` in `app/config.py` if needed

### Out of Memory
- Use `tinyllama` instead of `phi3`
- Reduce `MAX_TOKENS` in `app/config.py`
- Close other applications

---

**Current Setup:** Using `phi3` for LLM and `nomic-embed-text` for embeddings (if available).






