import os
import glob
import json
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from predict import CatDogPredictor

st.set_page_config(
    page_title="Cat vs Dog AI Detector",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .cat-badge {
        background: linear-gradient(135deg, #FF6B6B, #EE5253);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .dog-badge {
        background: linear-gradient(135deg, #4834D4, #686DE0);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .accuracy-highlight {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "output/best_cat_dog_model.pth"
HISTORY_PATH = "output/training_history.json"
PLOT_PATH = "output/training_results.png"
ANIMALS_DIR = "archive/animals"

@st.cache_resource
def load_predictor():
    if os.path.exists(MODEL_PATH):
        try:
            return CatDogPredictor(checkpoint_path=MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

predictor = load_predictor()

# Sidebar
st.sidebar.image("https://images.unsplash.com/photo-1548767797-d8c844163c4c?w=400&q=80", use_container_width=True)
st.sidebar.title("🐾 AI Model Controls")

# Check dataset stats
cat_files = glob.glob(os.path.join(ANIMALS_DIR, "cat", "*"))
dog_files = glob.glob(os.path.join(ANIMALS_DIR, "dog", "*"))
total_images = len(cat_files) + len(dog_files)

st.sidebar.markdown(f"""
### 📁 Dataset Info
- **Dataset Path**: `{ANIMALS_DIR}`
- **🐱 Cat Images**: {len(cat_files)}
- **🐶 Dog Images**: {len(dog_files)}
- **Total Images**: {total_images}
""")

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        meta = json.load(f)
    st.sidebar.markdown(f"""
    ### ⚙️ Trained Model Info
    - **Architecture**: `{meta.get('backbone', 'MobileNetV2').upper()}`
    - **Best Val Accuracy**: **{meta.get('best_val_acc', 0)*100:.2f}%**
    - **Final Train Accuracy**: **{meta.get('final_train_acc', 0)*100:.2f}%**
    """)
else:
    st.sidebar.warning("Model not yet trained or output files missing.")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Based on the transfer learning workflow from the notebook image using deep convolutional feature extraction and early stopping.")

# Main layout
st.markdown('<div class="main-title">🐱 Cat vs Dog AI Classifier & Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Computer Vision model powered by PyTorch, Transfer Learning, and Early Stopping.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔍 Live Single Image Predictor", "📁 Batch Prediction", "📊 Model Performance & Training Curves", "ℹ️ Architecture & Workflow"])

# TAB 1: Single Image
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Input Image")
        input_mode = st.radio("Choose Input Method:", ["Upload Image", "Pick Sample Image", "Use Camera"], horizontal=True)

        selected_image = None

        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload a Cat or Dog photo", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")

        elif input_mode == "Pick Sample Image":
            sample_options = []
            if cat_files:
                sample_options.extend(cat_files[:4])
            if dog_files:
                sample_options.extend(dog_files[:4])

            if sample_options:
                sample_names = [f"{os.path.basename(os.path.dirname(p)).upper()} - {os.path.basename(p)}" for p in sample_options]
                chosen_idx = st.selectbox("Select sample test image:", range(len(sample_options)), format_func=lambda i: sample_names[i])
                selected_image = Image.open(sample_options[chosen_idx]).convert("RGB")
            else:
                st.warning("No sample images found in archive/animals.")

        elif input_mode == "Use Camera":
            camera_pic = st.camera_input("Take a photo of your pet")
            if camera_pic is not None:
                selected_image = Image.open(camera_pic).convert("RGB")

        if selected_image is not None:
            st.image(selected_image, caption="Selected Image for Classification", use_container_width=True)

    with col2:
        st.subheader("2. AI Prediction Result")

        if selected_image is None:
            st.info("👈 Please upload or select an image on the left to see predictions.")
        else:
            if predictor is None:
                st.error("Model checkpoint not found. Please train the model first by running `python train.py`.")
            else:
                with st.spinner("Analyzing image features..."):
                    result = predictor.predict(selected_image)

                pred_class = result["predicted_class"].upper()
                confidence = result["confidence"] * 100
                is_cat = result["is_cat"]

                # Result banner
                if is_cat:
                    st.success(f"### 🐱 Detection: **CAT**")
                else:
                    st.info(f"### 🐶 Detection: **DOG**")

                st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                st.progress(float(result["confidence"]))

                st.markdown("#### Class Probabilities Breakdown:")
                prob_cat = result["probabilities"].get("cat", 0.0) * 100
                prob_dog = result["probabilities"].get("dog", 0.0) * 100

                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.metric(label="🐱 Cat Probability", value=f"{prob_cat:.2f}%")
                    st.progress(prob_cat / 100.0)
                with pcol2:
                    st.metric(label="🐶 Dog Probability", value=f"{prob_dog:.2f}%")
                    st.progress(prob_dog / 100.0)

                with st.expander("🔬 View Detailed Raw Output"):
                    st.json(result)

# TAB 2: Batch Prediction
with tabs[1]:
    st.subheader("📁 Batch Image Prediction")
    st.write("Upload multiple images at once to classify them in a single batch.")
    batch_files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    if batch_files and predictor:
        results_list = []
        progress_bar = st.progress(0.0)

        for idx, file in enumerate(batch_files):
            img = Image.open(file).convert("RGB")
            res = predictor.predict(img)
            results_list.append({
                "Filename": file.name,
                "Prediction": res["predicted_class"].upper(),
                "Confidence": f"{res['confidence']*100:.2f}%",
                "Cat %": f"{res['probabilities'].get('cat', 0)*100:.2f}%",
                "Dog %": f"{res['probabilities'].get('dog', 0)*100:.2f}%"
            })
            progress_bar.progress((idx + 1) / len(batch_files))

        df = pd.DataFrame(results_list)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="cat_dog_batch_predictions.csv",
            mime="text/csv"
        )
    elif not predictor:
        st.warning("Please train the model first to enable predictions.")

# TAB 3: Model Performance & Curves
with tabs[2]:
    st.subheader("📊 Model Performance & Training Curves")
    st.write("Direct visual comparison of Training and Validation Accuracy & Loss curves, replicating the notebook output shown in the project image.")

    if os.path.exists(PLOT_PATH):
        st.image(PLOT_PATH, caption="Training and Validation Accuracy and Loss (Generated during model training)", use_container_width=True)
    else:
        st.warning("Training plot not found. Run `python train.py` to generate the curves.")

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            data = json.load(f)

        hist = data.get("history", {})
        epochs = list(range(1, len(hist.get("train_acc", [])) + 1))

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric(label="🏆 Best Validation Accuracy", value=f"{data.get('best_val_acc', 0)*100:.2f}%")
        with mcol2:
            st.metric(label="🎯 Final Training Accuracy", value=f"{data.get('final_train_acc', 0)*100:.2f}%")
        with mcol3:
            st.metric(label="📉 Best Validation Loss", value=f"{min(hist.get('val_loss', [0])):.4f}")

        history_df = pd.DataFrame({
            "Epoch": epochs,
            "Train Accuracy": hist.get("train_acc", []),
            "Val Accuracy": hist.get("val_acc", []),
            "Train Loss": hist.get("train_loss", []),
            "Val Loss": hist.get("val_loss", [])
        })
        st.markdown("#### Epoch-by-Epoch History Table:")
        st.dataframe(history_df, use_container_width=True)

# TAB 4: Architecture & Workflow
with tabs[3]:
    st.subheader("ℹ️ Project Workflow & Deep Learning Pipeline")
    st.markdown("""
    This project is engineered based on the notebook shown in your reference image:
    
    1. **Importing Libraries**:
       - PyTorch, Torchvision, Pillow, Matplotlib, Pandas, NumPy, Streamlit.
    2. **The Dataset into Training and Testing Sets**:
       - Loads `archive/animals/cat` (500 images) and `archive/animals/dog` (500 images).
       - Automatically partitions with an 80/20 train-validation split.
    3. **Loading and Preprocessing Data**:
       - Resizes inputs to `(224, 224, 3)`.
       - Applies data augmentation (Random Horizontal Flip, Rotation, Color Jitter).
       - ImageNet standard normalization: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.
    4. **Building the Transfer Learning Model**:
       - Convolutional backbone as feature extractor (pre-trained on ImageNet).
       - Base layers frozen to retain spatial visual features.
       - Classifier head: `Flatten` $\\rightarrow$ `Linear(256) + ReLU` $\\rightarrow$ `Dropout(0.5)` $\\rightarrow$ `Linear(2)`.
    5. **Compiling the Model**:
       - Loss function: Cross Entropy Loss.
       - Optimizer: Adam optimizer with learning rate `0.001`.
    6. **Training with Early Stopping**:
       - Monitors validation loss with `patience=3` to prevent overfitting.
       - Automatically restores the best model weights.
    7. **Visualizing Results**:
       - Generates side-by-side plots for Training/Validation Accuracy and Loss.
    """)
