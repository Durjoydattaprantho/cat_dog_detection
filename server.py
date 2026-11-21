import os
import glob
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from detector import CatDogDetector

app = FastAPI(
    title="Cat vs Dog Detection & Localization API",
    description="Deep Learning API for Cat/Dog classification, bounding box localization, and Grad-CAM saliency heatmaps",
    version="2.0.0"
)

# Enable CORS for file:// and cross-origin browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

def get_detector():
    global detector
    if detector is None:
        checkpoint_path = "output/best_cat_dog_model.pth"
        if os.path.exists(checkpoint_path):
            detector = CatDogDetector(checkpoint_path=checkpoint_path)
        else:
            raise HTTPException(status_code=500, detail="Model checkpoint not found. Please train first.")
    return detector

@app.on_event("startup")
def startup_event():
    try:
        get_detector()
        print("CatDogDetector successfully initialized.")
    except Exception as e:
        print(f"Warning at startup: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = "index.html" if os.path.exists("index.html") else os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index template not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/samples_cache.json")
async def serve_samples_cache():
    if os.path.exists("samples_cache.json"):
        return FileResponse("samples_cache.json")
    raise HTTPException(status_code=404, detail="Cache not found")

@app.post("/api/detect")
async def detect_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.35)
):
    det = get_detector()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = det.detect(contents, box_threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

class SampleDetectRequest(BaseModel):
    filepath: str
    threshold: Optional[float] = 0.35

@app.post("/api/detect-sample")
async def detect_sample(req: SampleDetectRequest):
    det = get_detector()
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail="Sample image file not found.")

    try:
        result = det.detect(req.filepath, box_threshold=req.threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/samples")
async def get_samples():
    animals_dir = "archive/animals"
    cat_files = sorted(glob.glob(os.path.join(animals_dir, "cat", "*.png")) + glob.glob(os.path.join(animals_dir, "cat", "*.jpg")))
    dog_files = sorted(glob.glob(os.path.join(animals_dir, "dog", "*.png")) + glob.glob(os.path.join(animals_dir, "dog", "*.jpg")))

    samples = []
    # 4 cats
    for path in cat_files[:4]:
        name = os.path.basename(path)
        samples.append({
            "name": name,
            "category": "cat",
            "filepath": path.replace("\\", "/"),
            "thumbnail_url": f"/api/thumbnail?path={path.replace('\\', '/')}"
        })
    # 4 dogs
    for path in dog_files[:4]:
        name = os.path.basename(path)
        samples.append({
            "name": name,
            "category": "dog",
            "filepath": path.replace("\\", "/"),
            "thumbnail_url": f"/api/thumbnail?path={path.replace('\\', '/')}"
        })

    return samples

@app.get("/api/thumbnail")
async def get_thumbnail(path: str):
    # Security check: ensure path is inside archive/animals
    clean_path = os.path.normpath(path)
    if not clean_path.startswith("archive"):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(clean_path)

@app.get("/api/stats")
async def get_stats():
    animals_dir = "archive/animals"
    cat_count = len(glob.glob(os.path.join(animals_dir, "cat", "*")))
    dog_count = len(glob.glob(os.path.join(animals_dir, "dog", "*")))
    return {
        "cat_count": cat_count,
        "dog_count": dog_count,
        "total": cat_count + dog_count,
        "model_available": os.path.exists("output/best_cat_dog_model.pth")
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
