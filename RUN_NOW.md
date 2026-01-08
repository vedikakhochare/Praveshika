# 🚀 Quick Start - Run the Chatbot NOW!

## ✅ Prerequisites Check
- ✅ Python 3.13.5 installed
- ✅ Ollama running with phi3, tinyllama models
- ✅ All Python packages installed

## 📝 Step-by-Step Instructions

### Step 1: Place Your Admission Data File

Put your admission document (`.docx` or `.txt`) in the `data/` folder:
```
data/
  └── admission_info.docx  (or any .docx/.txt file)
```

**If you don't have a file yet**, create a test file:
```powershell
# Create a simple test file
echo "Eligibility: Minimum 60% in 12th grade. Documents required: Marksheet, ID proof. Fee: Rs. 50,000 per year." > data\test_admission.txt
```

### Step 2: Start the Server

Run this command:
```powershell
python run.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Keep this terminal window open!**

### Step 3: Load Your Data (First Time Only)

Open a **NEW terminal/PowerShell window** and run:

**Option A: Using PowerShell (Recommended for Windows)**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/load-data" -Method POST -ContentType "application/json" -Body "{}"
```

**Option B: Using the API Docs (Easiest)**
1. Open browser: http://localhost:8000/docs
2. Find `/load-data` endpoint
3. Click "Try it out"
4. Click "Execute"
5. Wait for response (may take 1-2 minutes)

### Step 4: Use the Chatbot!

Open your browser and go to:
```
http://localhost:8000
```

Start asking questions like:
- "What is the eligibility criteria?"
- "What documents are required?"
- "What is the fee structure?"

---

## 🎯 Quick Commands

### Start Server
```powershell
python run.py
```

### Check Status
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/status" -Method GET
```

### Load Data
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/load-data" -Method POST -ContentType "application/json" -Body "{}"
```

### Stop Server
Press `Ctrl+C` in the terminal where server is running

---

## ⚠️ Troubleshooting

### Port 8000 already in use?
```powershell
# Use a different port
uvicorn app.main:app --port 8001
```

### "No data file found"?
- Make sure you have a `.docx` or `.txt` file in the `data/` folder
- Check the file name

### "Cannot connect to Ollama"?
- Make sure Ollama is running: `ollama list`
- Restart Ollama if needed

---

**That's it! You're ready to go! 🎉**






