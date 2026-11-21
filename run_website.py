"""
Launch script for the Cat vs Dog AI Detection & Localization Web Application.
Starts the FastAPI server with modern HTML5/JS Studio.
"""
import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("🐾 CAT VS DOG AI DETECTION & LOCALIZATION STUDIO")
    print("=" * 65)
    print("Serving on: http://127.0.0.1:8000")
    print("Swagger API Docs: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop.")
    print("=" * 65)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
