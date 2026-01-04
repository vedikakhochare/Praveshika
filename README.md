# Admission Assistant Chatbot

A **local-only, RAG-based chatbot** for engineering college admissions using LLaMA 3.2 via Ollama.

## 🎯 Project Overview

This chatbot answers admission-related queries using **only** official admission data. It uses:
- **LLaMA 3.2** (via Ollama) for text generation
- **nomic-embed-text** (via Ollama) for embeddings
- **FAISS** for local vector storage
- **FastAPI** for the backend API
- **RAG architecture** to prevent hallucinations

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Install required models:
     **Required LLM Model (choose one):**
     ```bash
     ollama pull phi3
     # OR (if you prefer a smaller model)
     ollama pull tinyllama
     ```
     
     **Optional (but recommended) Embedding Model:**
     ```bash
     ollama pull nomic-embed-text
     ```
     
     **Note:** The system is configured to use `phi3` by default. If you want to use `tinyllama` instead, set the environment variable `LLM_MODEL=tinyllama`. If you don't have `nomic-embed-text`, the system will attempt to use your LLM model for embeddings (though this is less optimal).

## 🚀 Installation

1. **Clone/Navigate to project directory:**
   ```bash
   cd admitbot
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place admission data file:**
   - Put your admission document (`.docx` or `.txt`) in the `data/` directory
   - Example: `data/admission_info.docx`

## 🏃 Running the Application

1. **Ensure Ollama is running:**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # If not running, start Ollama service
   # (Usually runs automatically after installation)
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Load admission data (first time only):**
   ```bash
   # Using curl
   curl -X POST "http://localhost:8000/load-data" \
        -H "Content-Type: application/json" \
        -d "{}"
   
   # Or specify a file
   curl -X POST "http://localhost:8000/load-data" \
        -H "Content-Type: application/json" \
        -d '{"filename": "admission_info.docx"}'
   ```

4. **Access the chatbot:**
   - Open browser: http://localhost:8000
   - Or use API docs: http://localhost:8000/docs

## 📁 Project Structure

```
admitbot/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── data_ingestion.py    # DOCX/TXT file reading
│   ├── text_chunking.py     # Text splitting for RAG
│   ├── embeddings.py        # Ollama embeddings client
│   ├── vector_store.py      # FAISS vector database
│   └── rag.py               # RAG pipeline
├── data/                    # Place admission documents here
├── vector_store/            # FAISS index storage (auto-created)
├── templates/
│   └── index.html           # Frontend UI
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 API Endpoints

### `POST /ask`
Ask a question to the chatbot.

**Request:**
```json
{
  "question": "What is the eligibility criteria for admission?"
}
```

**Response:**
```json
{
  "answer": "Based on the official admission data..."
}
```

### `POST /load-data`
Load and process admission data file.

**Request:**
```json
{
  "filename": "admission_info.docx"  // Optional, auto-detects if omitted
}
```

**Response:**
```json
{
  "message": "Data loaded successfully",
  "chunks_processed": 150,
  "filename": "admission_info.docx"
}
```

### `GET /status`
Get system status and statistics.

**Response:**
```json
{
  "vector_store_loaded": true,
  "vector_store_stats": {
    "count": 150,
    "dimension": 768
  },
  "models_available": {
    "llm_model": true,
    "embedding_model": true,
    "ollama_available": true
  },
  "available_data_files": ["admission_info.docx"]
}
```

### `GET /health`
Health check endpoint.

## 🧪 Testing

### Test with curl:

```bash
# Check status
curl http://localhost:8000/status

# Ask a question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the required documents?"}'
```

### Test with Python:

```python
import requests

# Ask question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is the fee structure?"}
)
print(response.json()["answer"])
```

## ⚙️ Configuration

Edit `app/config.py` to adjust:

- **Temperature**: Lower = more deterministic (default: 0.2)
- **Chunk size**: Text chunk size for RAG (default: 500 chars)
- **Top K**: Number of chunks to retrieve (default: 3)
- **Ollama URL**: If Ollama runs on different port (default: http://localhost:11434)

## 🐛 Common Issues & Fixes

### 1. **Ollama Connection Error**
```
Error: Cannot connect to Ollama at http://localhost:11434
```

**Fix:**
- Ensure Ollama is running: `ollama list`
- Check if models are installed: `ollama pull phi3` (or `tinyllama`) and optionally `ollama pull nomic-embed-text`
- Verify Ollama is accessible: `curl http://localhost:11434/api/tags`

### 2. **No Data File Found**
```
FileNotFoundError: No admission data file found in data/
```

**Fix:**
- Place a `.docx` or `.txt` file in the `data/` directory
- Or specify filename in `/load-data` request

### 3. **FAISS Import Error**
```
ModuleNotFoundError: No module named 'faiss'
```

**Fix:**
- Install FAISS: `pip install faiss-cpu`
- On Linux, you may need: `pip install faiss-cpu --no-cache-dir`

### 4. **Empty Responses**
If chatbot returns "I do not have this information":

- Check if data was loaded: `GET /status`
- Verify the question is related to admission data
- Try rephrasing the question
- Check if relevant information exists in the source document

### 5. **Slow Response Times**
- Reduce `TOP_K` in `config.py` (fewer chunks to retrieve)
- Reduce `CHUNK_SIZE` for faster embedding generation
- Ensure Ollama is running locally (not remote)

### 6. **Port Already in Use**
```
Error: Address already in use
```

**Fix:**
- Use different port: `uvicorn app.main:app --port 8001`
- Or stop the process using port 8000

## 📝 How It Works

1. **Data Ingestion**: Reads DOCX/TXT files and extracts text
2. **Chunking**: Splits text into overlapping chunks (preserves context)
3. **Embedding**: Generates vector embeddings using Ollama
4. **Storage**: Stores embeddings in FAISS vector database
5. **Retrieval**: Finds relevant chunks for user queries
6. **Generation**: Uses LLaMA 3.2 to generate answers from retrieved context
7. **Response**: Returns grounded, accurate answers

## 🔒 Key Features

- ✅ **Local-only**: No cloud services, complete privacy
- ✅ **No Hallucinations**: Answers only from admission data
- ✅ **Low Temperature**: Deterministic, accurate responses
- ✅ **RAG Architecture**: Retrieval + Generation for accuracy
- ✅ **Persistent Storage**: FAISS index saved to disk
- ✅ **Simple UI**: Clean, modern web interface

## 📄 License

This project is designed for educational and institutional use.

## 🤝 Support

For issues or questions:
1. Check the "Common Issues & Fixes" section above
2. Verify Ollama models are installed
3. Check `/status` endpoint for system health
4. Review logs for detailed error messages

---

**Built with ❤️ for engineering college admissions**

