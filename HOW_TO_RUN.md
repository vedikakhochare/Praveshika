# How to Run the Admission Assistant Chatbot

## Prerequisites Check

First, ensure you have:
1. ✅ Python 3.8+ installed
2. ✅ Ollama installed and running
3. ✅ Required Ollama models installed

## Step-by-Step Instructions

### Step 1: Verify Ollama is Running

Open a terminal and check:
```bash
ollama list
```

If Ollama is not installed, download from: https://ollama.ai

### Step 2: Install Required Ollama Models

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

**Note:** The system is configured to use `phi3` by default. If you want to use `tinyllama` instead, you can set the environment variable:
```bash
# Windows PowerShell
$env:LLM_MODEL="tinyllama"

# Linux/Mac
export LLM_MODEL="tinyllama"
```

If you don't have `nomic-embed-text`, the system will attempt to use your LLM model for embeddings (though this is less optimal).

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Place Your Admission Data File

Copy your admission document (`.docx` or `.txt`) to the `data/` folder:
```
data/
  └── admission_info.docx  (or .txt)
```

### Step 5: (Optional) Run Setup Check

Verify everything is configured correctly:
```bash
python setup.py
```

### Step 6: Start the Server

**Option A: Using the run script (Recommended)**
```bash
python run.py
```

**Option B: Using uvicorn directly**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 7: Load Admission Data (First Time Only)

Open a **new terminal** (keep server running) and run:

**Using curl:**
```bash
curl -X POST "http://localhost:8000/load-data" -H "Content-Type: application/json" -d "{}"
```

**Using PowerShell (Windows):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/load-data" -Method POST -ContentType "application/json" -Body "{}"
```

**Or use the API docs:**
- Open: http://localhost:8000/docs
- Click on `/load-data` endpoint
- Click "Try it out"
- Click "Execute"

Wait for the response - this may take 1-2 minutes depending on document size.

### Step 8: Access the Chatbot

Open your browser and go to:
```
http://localhost:8000
```

You should see the chatbot interface!

### Step 9: Start Asking Questions

Try questions like:
- "What is the eligibility criteria?"
- "What documents are required?"
- "What is the fee structure?"
- "Explain the admission process"

---

## Quick Commands Reference

### Check Server Status
```bash
curl http://localhost:8000/status
```

### Ask a Question via API
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is the eligibility criteria?\"}"
```

### Check Health
```bash
curl http://localhost:8000/health
```

---

## Troubleshooting

### Server won't start?
- Check if port 8000 is already in use
- Try: `uvicorn app.main:app --port 8001`

### "Cannot connect to Ollama" error?
- Ensure Ollama is running: `ollama list`
- Check if models are installed: `ollama list`

### "No data file found" error?
- Place a `.docx` or `.txt` file in `data/` folder
- Check file name matches what you're trying to load

### Import errors?
- Run: `pip install -r requirements.txt`
- Ensure you're in the project directory

---

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

---

**That's it! You're ready to use the chatbot.** 🎉

