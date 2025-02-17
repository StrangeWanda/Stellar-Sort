import os
import cv2
import roboflow
import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from n import Galaxy10CNN
from helpIm import *

# --- Initialize Roboflow (Object Detection) ---
rf = roboflow.Roboflow(api_key="eupf89ahjno75Ip8E94w")
project = rf.workspace().project("galexymokinj")
rf_model = project.version("1").model  # Renamed from 'model' to 'rf_model' to avoid conflict

# --- Load Classification Model ---
def get_latest_checkpoint(checkpoint_dir="Inbw"):
    # checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    return "galaxy10_final.pth"

def load_model():
    checkpoint_path = get_latest_checkpoint()
    if not checkpoint_path:
        st.error("🚨 No model checkpoint found!")
        return None
    classifier_model = Galaxy10CNN()  # Renamed to avoid conflict
    classifier_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    classifier_model.eval()
    return classifier_model

# --- Preprocessing & Prediction Functions ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

def predict(classifier_model, img_tensor):
    with torch.no_grad():
        output = classifier_model(img_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# --- Galaxy Classes ---
galaxy_classes = {
    0: "🌌 Disk, Face-on, No Spiral",
    1: "🔵 Smooth, Completely Round",
    2: "🌀 Smooth, In Between Round",
    3: "📏 Smooth, Cigar Shaped",
    4: "🎯 Edge-on, Rounded Bulge",
    5: "🟦 Edge-on, Boxy Bulge",
    6: "📜 Edge-on, No Bulge",
    7: "⚡ Spiral, Tight",
    8: "✨ Spiral, Medium",
    9: "🌠 Spiral, Loose"
}

# --- Image Processing (Cutter) ---
def process_with_cutter(image_path, confidence, max_size=(256, 256)):
    """Run galaxy detection on an image and extract detected objects with resized outputs."""
    output_dir = "extracted_objects"
    os.makedirs(output_dir, exist_ok=True)

    # Run inference with Roboflow object detection model
    prediction = rf_model.predict(image_path, confidence=confidence)
    predictions = prediction.json().get('predictions', [])

    # Load original image
    image = cv2.imread(image_path)
    extracted_images = []

    if predictions:
        for i, pred in enumerate(predictions):
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cropped_image = image[y1:y2, x1:x2]

            # Resize to a manageable size
            cropped_image = cv2.resize(cropped_image, max_size, interpolation=cv2.INTER_AREA)

            output_path = os.path.join(output_dir, f"object_{i + 1}.jpg")
            cv2.imwrite(output_path, cropped_image)
            extracted_images.append(output_path)

    return extracted_images if extracted_images else [image_path]  # Return cropped images if available, else original image

# --- Streamlit UI ---

# Page Title
st.set_page_config(page_title="Stellar Sort", page_icon="🌌", layout="wide")

st.markdown("<h1 style='text-align: center;'>🌟 StellarSort Pipeline 🌟</h1>", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/0d/Hubble_ultra_deep_field_high_rez_edit1.jpg", use_container_width=True)
st.sidebar.markdown('''## About ## 
Exploring the cosmos has never been easier. Our platform leverages cutting-edge machine learning to automate galaxy classification, helping astronomers accelerate their research in cosmology.

Using advanced object detection with YOLO, we first segment images containing multiple galaxies, isolating each one for analysis. Then, our classification model categorizes them into elliptical, spiral, or irregular types based on their shape, size, and spectral features—enabling faster, more efficient galaxy identification.

Powered by data from sources like the Sloan Digital Sky Survey (SDSS), this project transforms raw astronomical observations into structured insights, making galaxy classification more accessible, accurate, and scalable.

Step into the future of space exploration—one galaxy at a time.''', unsafe_allow_html=True)

# Confidence Threshold Slider (Now in Main UI)
confidence_threshold = st.slider(
    "🔍 Set Detection Confidence Threshold",
    min_value=0.01,
    max_value=1.0,
    value=0.04,
    step=0.01
)

# File Upload
uploaded_files = st.file_uploader("📤 Upload a galaxy image", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# Classify Button
if uploaded_files and st.button("🔍 Classify Galaxies"):
    classifier_model = load_model()  # Load the classifier model separately
    if classifier_model:
        for uploaded_file in uploaded_files:
            image_path = f"temp_{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(image_path, caption="📷 Original Image", width=400)  # Controlled display size

            # Process with Cutter using selected confidence
            st.info(f"🔍 Detecting galaxies (Confidence: {confidence_threshold * 100:.0f}%)...")
            extracted_images = process_with_cutter(image_path, confidence_threshold)

            # Perform Classification
            st.info(f"🧠 Classifying {len(extracted_images)} image(s)...")
            for img_path in extracted_images:
                image = Image.open(img_path)
                img_tensor = preprocess_image(image)
                predicted_class = predict(classifier_model, img_tensor)
                galaxy_type = galaxy_classes.get(predicted_class, "❓ Unknown")
                
                st.image(image, caption=f"🔭 Classified as: **{galaxy_type}**", width=300)  # Controlled size
            
        st.success("✅ All images processed successfully!")
