# Architecture Overview

## System Architecture

The Admission Assistant Chatbot uses a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, grounded answers from official admission data.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (HTML/JavaScript)                         │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP Requests
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  /ask        │  │ /load-data   │  │ /status      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
└─────────┼──────────────────┼────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Query Embedding (Ollama nomic-embed-text)       │  │
│  │  2. Vector Search (FAISS)                            │  │
│  │  3. Context Retrieval (Top-K chunks)                 │  │
│  │  4. Answer Generation (LLaMA 3.2)                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────┬──────────────────┬────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────┐  ┌─────────────────────┐
│   FAISS Vector DB   │  │   Ollama Service    │
│   (Local Storage)   │  │  - LLaMA 3.2       │
│                     │  │  - nomic-embed-text │
└─────────────────────┘  └─────────────────────┘
```

## Component Details

### 1. Data Ingestion (`app/data_ingestion.py`)
**Purpose**: Extract text from admission documents

**Process**:
- Reads DOCX files using `python-docx`
- Reads TXT files with UTF-8 encoding
- Extracts text from paragraphs and tables
- Returns clean text for processing

**Why**: Converts structured documents into processable text format.

---

### 2. Text Chunking (`app/text_chunking.py`)
**Purpose**: Split large documents into manageable chunks

**Process**:
- Splits text into chunks of ~500 characters
- Maintains 50-character overlap between chunks
- Prefers sentence/paragraph boundaries
- Preserves context continuity

**Why**: 
- LLMs have context window limits
- Smaller chunks improve retrieval precision
- Overlap prevents information loss at boundaries

---

### 3. Embeddings (`app/embeddings.py`)
**Purpose**: Convert text into numerical vectors

**Process**:
- Uses Ollama's `nomic-embed-text` model
- Generates 768-dimensional vectors
- Captures semantic meaning of text
- Enables similarity search

**Why**:
- Similar texts have similar vectors
- Enables fast semantic search
- Foundation for RAG retrieval

---

### 4. Vector Store (`app/vector_store.py`)
**Purpose**: Store and search document embeddings

**Process**:
- Uses FAISS (Facebook AI Similarity Search)
- Stores normalized embeddings
- Performs fast L2 distance search
- Converts distances to similarity scores
- Persists to disk for reuse

**Why**:
- FAISS is optimized for similarity search
- Local storage = privacy + speed
- Persistence avoids re-embedding

---

### 5. RAG Pipeline (`app/rag.py`)
**Purpose**: Retrieve context and generate answers

**Process**:
1. **Retrieve**: 
   - Embed user query
   - Search vector store for top-K similar chunks
   - Filter by similarity threshold

2. **Generate**:
   - Format prompt with retrieved context
   - Send to LLaMA 3.2 via Ollama
   - Use low temperature (0.2) for accuracy
   - Return generated answer

**Why RAG**:
- **Retrieval** ensures answers come from admission data
- **Generation** creates natural language responses
- **Together** = accurate, grounded answers without hallucinations

---

### 6. FastAPI Backend (`app/main.py`)
**Purpose**: Expose REST API and serve frontend

**Endpoints**:
- `POST /ask`: Main chatbot endpoint
- `POST /load-data`: Load and process admission documents
- `GET /status`: System health and statistics
- `GET /health`: Simple health check
- `GET /`: Serve frontend HTML

**Why FastAPI**:
- Modern, fast Python web framework
- Automatic API documentation
- Type validation with Pydantic
- Easy to deploy

---

### 7. Frontend (`templates/index.html`)
**Purpose**: User-friendly web interface

**Features**:
- Clean, modern UI
- Real-time status indicators
- Chat interface
- Loading states
- Error handling

**Why**: Provides accessible interface for end users.

---

## Data Flow

### Initialization Flow:
```
1. Start FastAPI server
2. Initialize vector store (load from disk if exists)
3. Initialize RAG pipeline
4. Server ready
```

### Data Loading Flow:
```
1. User uploads/places document in data/
2. POST /load-data called
3. Extract text from document
4. Chunk text intelligently
5. Generate embeddings via Ollama
6. Store in FAISS
7. Save to disk
8. Ready for queries
```

### Query Flow:
```
1. User asks question via UI
2. POST /ask with question
3. RAG pipeline:
   a. Embed query
   b. Search FAISS for similar chunks
   c. Retrieve top-K chunks
   d. Format prompt with context
   e. Generate answer via LLaMA 3.2
4. Return answer to user
```

---

## Key Design Decisions

### 1. **Low Temperature (0.2)**
- **Why**: Reduces randomness, increases accuracy
- **Effect**: More deterministic, factual responses

### 2. **Chunk Size (500 chars)**
- **Why**: Balance between context and precision
- **Effect**: Relevant chunks without overwhelming context

### 3. **Top-K = 3**
- **Why**: Enough context without diluting relevance
- **Effect**: Focused, accurate answers

### 4. **Similarity Threshold (0.5)**
- **Why**: Filter out irrelevant chunks
- **Effect**: Only use truly relevant context

### 5. **FAISS L2 Normalization**
- **Why**: Enables cosine similarity via L2 distance
- **Effect**: Better semantic matching

### 6. **Strict Prompt Template**
- **Why**: Enforce "only from context" behavior
- **Effect**: Prevents hallucinations

---

## Security & Privacy

- ✅ **Local-only**: No data leaves the machine
- ✅ **No cloud services**: Complete privacy
- ✅ **No external APIs**: No data sharing
- ✅ **Offline capable**: Works without internet (after setup)

---

## Performance Considerations

1. **Embedding Generation**: Batch processing for multiple chunks
2. **FAISS Index**: Fast similarity search (milliseconds)
3. **Ollama Local**: No network latency for LLM calls
4. **Persistent Storage**: Avoid re-embedding on restart
5. **Caching**: Vector store persists between sessions

---

## Scalability

**Current Design**: Optimized for single-college deployment

**For Multi-College**:
- Separate vector stores per college
- Route queries based on college selection
- Shared infrastructure, isolated data

**For Higher Load**:
- Add Redis caching for frequent queries
- Use FAISS GPU for faster search
- Load balance multiple Ollama instances

---

## Error Handling

1. **Ollama Connection**: Clear error messages, fallback suggestions
2. **Missing Data**: Graceful degradation, helpful messages
3. **Invalid Queries**: Validation, user-friendly errors
4. **Model Availability**: Status checks, installation guidance

---

## Testing Strategy

1. **Unit Tests**: Each module independently
2. **Integration Tests**: End-to-end RAG pipeline
3. **API Tests**: FastAPI endpoints
4. **E2E Tests**: Full user flow

---

## Future Enhancements

1. **Multi-language Support**: Translate queries/responses
2. **Conversation History**: Context-aware follow-ups
3. **Confidence Scores**: Show answer certainty
4. **Source Citations**: Link answers to document sections
5. **Admin Dashboard**: Monitor queries, update data

---

**Architecture designed for: Accuracy, Privacy, and Reliability**





