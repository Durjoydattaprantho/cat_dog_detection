# 🐾 Cat vs Dog AI Detection & Localization Studio

A comprehensive Deep Learning Computer Vision project for **Cat and Dog Detection & Localization**, featuring two complete web applications, transfer learning with PyTorch, Early Stopping, and Explainable AI (Grad-CAM heatmaps & bounding box localization).

---

## 🌟 Live Demo on GitHub Pages

Experience the interactive detection studio live in your browser:
👉 **[https://durjoydattaprantho.github.io/cat_dog_detection/](https://durjoydattaprantho.github.io/cat_dog_detection/)**

*Includes instant dataset testing for 8 pre-computed pets (bounding boxes, Grad-CAM attention heatmaps, and confidence telemetry), with support for custom live backends.*

### ⚡ How to Enable GitHub Pages (1-Minute Setup):
1. Open your repository on GitHub: [Durjoydattaprantho/cat_dog_detection](https://github.com/Durjoydattaprantho/cat_dog_detection)
2. Go to **Settings** $\rightarrow$ **Pages** (in the left sidebar).
3. Under **Build and deployment** $\rightarrow$ **Source**, choose **Deploy from a branch**.
4. Set Branch to **`main`** and folder to **`/(root)`**, then click **Save**.
5. Within 1-2 minutes, your live site will be accessible at:
   `https://durjoydattaprantho.github.io/cat_dog_detection/`

---

## 🚀 Two Full Web Application Experiences

### 🌐 Type 1: Modern Full-Stack AI Detection Studio (FastAPI + HTML5/Tailwind/JS)
Features real-time **Object Localization with Bounding Boxes**, **Grad-CAM Attention Heatmaps**, **Live Webcam Stream**, and **Dataset Sample Explorer**:
```bash
python run_website.py
```
- **Web Interface:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive REST API (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Highlights:**
  - 🎯 **Bounding Box Object Detection:** Pinpoints the exact region of the cat or dog with cyber brackets, coordinates `(X, Y, W, H)`, and confidence badge.
  - 🔥 **Explainable AI (Grad-CAM):** Visualizes convolutional neural activations highlighting ears, eyes, whiskers, and body morphology.
  - 📷 **Live Web Camera Mode:** Detects pets directly from your device camera.
  - 🖼️ **Dataset Sample Explorer:** 1-click testing of 8 built-in images from `archive/animals`.
  - ⚡ **Real-Time Telemetry:** Latency timer, probability breakdown, and sensitivity threshold slider.

---

### 📊 Type 2: Streamlit Data Science Web Dashboard
Features dashboard analytics, multi-image batch classification with CSV export, and training history curves:
```bash
streamlit run app.py
```
- **Web Interface:** [http://localhost:8501](http://localhost:8501)
- **Highlights:**
  - 🔍 **Live Single Image Classifier:** Upload or pick sample pets with probability gauges.
  - 📁 **Batch Processing:** Upload multiple images and export predictions to CSV.
  - 📈 **Performance Dashboard:** Visualizes training/validation accuracy and loss curves matching the reference notebook.

---

## 🧠 Model Architecture & Training Results

- **Dataset:** 1,000 images in `archive/animals` (500 Cat, 500 Dog).
- **Split:** 80% Train (800 images), 20% Validation (200 images).
- **Backbone:** Pre-trained Convolutional Base (`MobileNetV2` / `VGG16`) with frozen feature extractors.
- **Classifier Head:** `Flatten` $\rightarrow$ `Linear(256) + ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(2)`.
- **Optimization:** Adam Optimizer, CrossEntropyLoss, and Early Stopping (`patience=3`).
- **Validation Accuracy:** **`100.00%`** | **Training Accuracy:** **`99.50%`**
- **Artifacts:**
  - Checkpoint: `output/best_cat_dog_model.pth`
  - Training Curves: `output/training_results.png`

---

## 🛠️ Command-Line Usage

### 1. Run Object Detection via Python
```bash
python -c "from detector import CatDogDetector; d = CatDogDetector(); print(d.detect('archive/animals/cat/00000-4122619873.png'))"
```

### 2. Run Classification via CLI
```bash
python predict.py --image "archive/animals/cat/00000-4122619873.png"
```

### 3. Retrain the Model
```bash
python train.py --epochs 5 --batch_size 32
```

---

## 📁 Project Structure
```
Imtiaz_ml_project/
├── archive/
│   └── animals/
│       ├── cat/                     # 500 Cat images
│       └── dog/                     # 500 Dog images
├── templates/
│   └── index.html                   # Modern Detection Studio Web UI
├── detector.py                      # Bounding box & Grad-CAM detection engine
├── server.py                        # FastAPI web server & REST API
├── run_website.py                   # 1-click launcher for FastAPI Web Studio
├── app.py                           # Streamlit Data Science Dashboard
├── dataset.py                       # Dataset loader & augmentations
├── model.py                         # Deep Learning architecture
├── train.py                         # Training pipeline with early stopping
├── predict.py                       # CLI inference helper
├── cat_dog_classification.ipynb     # Jupyter Notebook matching the image
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```
