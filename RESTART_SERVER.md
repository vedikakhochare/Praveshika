# Server Restart Required

The code changes require a server restart to take effect.

## To Fix the Retrieval Issue:

1. **Stop the current server:**
   - Find the terminal where `python run.py` is running
   - Press `Ctrl+C` to stop it

2. **Restart the server:**
   ```powershell
   python run.py
   ```

3. **Reload the data:**
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/load-data" -Method POST -ContentType "application/json" -Body '{"filename": "kjsit_admission_data.txt"}'
   ```

4. **Test again:**
   - Open http://localhost:8000
   - Try: "What are the cutoff marks for Computer Engineering?"

## Changes Made:
- ✅ Lowered similarity threshold from 0.5 to 0.3
- ✅ Increased chunk size from 500 to 800 characters
- ✅ Increased TOP_K from 3 to 5
- ✅ Modified search to always return top-k results

These changes should improve retrieval of cutoff information.






