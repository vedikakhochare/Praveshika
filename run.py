"""
Quick start script for Admission Assistant Chatbot.

Usage: python run.py
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Admission Assistant Chatbot - Starting Server")
    print("=" * 60)
    print("\nServer will start at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )





