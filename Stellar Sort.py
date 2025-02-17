import streamlit as st
import os
import torch
import time
import torchvision.transforms as transforms
from PIL import Image
from n import Galaxy10CNN  # Import your trained model class
from helpIm import *

# --- Custom CSS for a Futuristic, Dark-Themed Look ---
st.markdown("""
    <style>
        /* Apply dark background to the entire app container */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0D0D0D, #1A1A1A) !important;
            color: #E0E0E0;
            font-family: 'Poppins', sans-serif;
        }
        /* Override default white background on main block container */
        .block-container {
            background: transparent;
        }
        /* Header Styles */
        .header-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 42px;
            text-align: center;
            color: #FFD700;
            text-shadow: 0 0 15px #FFD700;
            animation: glow 2s infinite alternate;
        }
        @keyframes glow {
            0% { text-shadow: 0 0 5px #FFD700; }
            100% { text-shadow: 0 0 20px #FFD700, 0 0 30px #FFA500; }
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 20px;
            text-align: center;
            color: #FF8C00;
        }
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(25, 25, 25, 0.9);
            color: #ffffff;
            border-right: 2px solid #FFD700;
            backdrop-filter: blur(10px);
        }
        .sidebar-text {
            font-size: 16px;
            color: #E0E0E0;
        }
        /* Button Styling */
        .stButton>button {
            background: linear-gradient(90deg, #FFD700, #FFA500);
            color: black;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 4px 10px rgba(255, 165, 0, 0.5);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #FFA500, #FF4500);
            transform: scale(1.05);
        }
        /* Footer Styling */
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            padding: 10px;
            color: #888888;
        }
        /* Upload Box Button Styling */
        .stFileUploader>div>div>div>button {
            background-color: #FF8C00 !important;
            color: black !important;
            border-radius: 10px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Functions for Model Handling ---
def get_latest_checkpoint(checkpoint_dir="Inbw"):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        st.error("🚨 No model checkpoint found!")
        return None
    latest_checkpoint = sorted(checkpoints)[-1]
    return "galaxy10_final.pth"
    return os.path.join(checkpoint_dir, latest_checkpoint)

def load_model():
    checkpoint_path = get_latest_checkpoint()
    if checkpoint_path is None:
        return None
    model = Galaxy10CNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# --- Galaxy Class Labels ---
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


# Header Section with Cosmic Animation
st.markdown("<div class='header-title'>🌟 StellarSort 🌟</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Sorting and classifying galaxies like a cosmic librarian</div>", unsafe_allow_html=True)

# Animated Space-Themed GIF
# st.markdown("""
# <div style="text-align: center;">
#   <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW40NHI0MGc1cjJzcDN3emN4NHRpMHQ4Y3NueTNjcmxsY3o3a254NiZlcD12MV9pbnRlcm5naWZfYnlfaWQmY3Q9Zw/3og0IFrHkIglEOg8Ba/giphy.gif" 
#        style="height:280px; width:auto; margin:auto; border-radius: 10px;"/>
# </div>
# """, unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/0d/Hubble_ultra_deep_field_high_rez_edit1.jpg", use_container_width=True)
st.sidebar.markdown("<div class='sidebar-text'><strong>About StellarSort 🚀</strong></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='sidebar-text'>This app uses a deep learning model trained on the <em>Galaxy10</em> dataset to classify galaxies. Simply upload an image, and let AI reveal the secrets of the cosmos! ✨</div>",
    unsafe_allow_html=True
)

# File Uploader
uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if uploaded_files:
    images = [Image.open(file) for file in uploaded_files]

    # Display uploaded images in a neat grid
    num_images = len(images)
    cols = st.columns(min(3, num_images))
    for col, image in zip(cols, images):
        col.image(image, caption="Uploaded Galaxy", use_container_width=True)

    # Classify Button
    if st.button("🔭 Classify Galaxies"):
        with st.spinner("🔍 Analyzing galaxies..."):
            time.sleep(2)  # Simulated processing time
            model = load_model()
            if model is None:
                st.error("🚨 Failed to load the model. Ensure training has been completed!")
            else:
                for image in images:
                    img_tensor = preprocess_image(image)
                    predicted_class = predict(model, img_tensor)
                    galaxy_type = galaxy_classes.get(predicted_class, "❓ Unknown")
                    st.image(image, caption=f"🔭 Classified as: **{galaxy_type}** / **{get_galaxy_type(predicted_class)}**", use_container_width=True)
        st.success("✅ Images analyzed successfully!")

# Footer
st.markdown("<div class='footer'>🔬 Developed on COSMANOVA | 🚀 Powered by PyTorch | 🌌 Explore the universe with AI!</div>", unsafe_allow_html=True)
