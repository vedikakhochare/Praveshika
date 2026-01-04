"""
Setup script for Admission Assistant Chatbot.

This script helps verify the environment and check prerequisites.
"""

import sys
import subprocess
import requests
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ollama():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
    except:
        pass
    
    print("❌ Ollama is not running or not accessible at http://localhost:11434")
    print("   Please install Ollama from https://ollama.ai and ensure it's running")
    return False


def check_ollama_models():
    """Check if required Ollama models are installed."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            # Check for LLM models (phi3 or tinyllama)
            phi3_installed = any("phi3" in name for name in model_names)
            tinyllama_installed = any("tinyllama" in name for name in model_names)
            llm_installed = phi3_installed or tinyllama_installed
            
            # Check for embedding model
            embed_installed = any("nomic-embed-text" in name for name in model_names)
            
            if phi3_installed:
                print("✓ Phi-3 model found")
            elif tinyllama_installed:
                print("✓ TinyLlama model found")
            else:
                print("❌ No LLM model found. Run: ollama pull phi3 (or ollama pull tinyllama)")
            
            if embed_installed:
                print("✓ nomic-embed-text model found (recommended)")
            else:
                print("⚠ nomic-embed-text model not found (optional, but recommended)")
                print("  Run: ollama pull nomic-embed-text for better embeddings")
            
            return llm_installed  # Only require LLM model, embedding is optional
    except:
        print("❌ Cannot check models (Ollama not accessible)")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    required = [
        "fastapi",
        "uvicorn",
        "requests",
        "docx",
        "faiss",
        "numpy",
        "pydantic"
    ]
    
    missing = []
    for package in required:
        try:
            if package == "docx":
                __import__("docx")
            elif package == "faiss":
                __import__("faiss")
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages: pip install -r requirements.txt")
        return False
    
    return True


def check_data_files():
    """Check if data files exist."""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("ℹ Created data/ directory")
    
    docx_files = list(data_dir.glob("*.docx"))
    txt_files = list(data_dir.glob("*.txt"))
    
    if docx_files or txt_files:
        print(f"✓ Found {len(docx_files) + len(txt_files)} data file(s)")
        return True
    else:
        print("ℹ No data files found in data/ directory")
        print("   Place your admission document (.docx or .txt) in data/ before loading")
        return True  # Not a blocker


def main():
    """Run all checks."""
    print("=" * 50)
    print("Admission Assistant Chatbot - Setup Check")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Dependencies", check_dependencies),
        ("Ollama Service", check_ollama),
        ("Ollama Models", check_ollama_models),
        ("Data Files", check_data_files),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        result = check_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All checks passed! You're ready to run the application.")
        print("\nNext steps:")
        print("  1. Start server: uvicorn app.main:app --reload")
        print("  2. Load data: POST /load-data")
        print("  3. Access UI: http://localhost:8000")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    print("=" * 50)


if __name__ == "__main__":
    main()

