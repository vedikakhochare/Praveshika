# Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Install Ollama
```bash
# Download from https://ollama.ai
# Install and start Ollama service
```

### Step 2: Install Models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Place Data File
```bash
# Copy your admission document to data/ directory
# Example: data/admission_info.docx
```

### Step 5: Run Setup Check (Optional)
```bash
python setup.py
```

### Step 6: Start Server
```bash
python run.py
# Or: uvicorn app.main:app --reload
```

### Step 7: Load Data
```bash
# Using curl:
curl -X POST "http://localhost:8000/load-data" \
     -H "Content-Type: application/json" \
     -d "{}"

# Or use the API docs at http://localhost:8000/docs
```

### Step 8: Use Chatbot
Open browser: **http://localhost:8000**

---

## 📋 Common Commands

### Check Status
```bash
curl http://localhost:8000/status
```

### Ask Question
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the eligibility criteria?"}'
```

### Check Health
```bash
curl http://localhost:8000/health
```

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not found | Install from https://ollama.ai |
| Models missing | Run `ollama pull llama3.2` and `ollama pull nomic-embed-text` |
| Port 8000 in use | Use `--port 8001` in uvicorn command |
| No data file | Place `.docx` or `.txt` in `data/` directory |
| Import errors | Run `pip install -r requirements.txt` |

---

## 📝 Example Questions

- "What is the eligibility criteria for admission?"
- "What documents are required?"
- "What is the fee structure?"
- "Explain the CAP admission process"
- "What are the cutoff marks for Computer Engineering?"

---

**Ready to go!** 🎉






