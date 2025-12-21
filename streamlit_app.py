import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Set Page Config
st.set_page_config(page_title="Custom YOLOv11 Detection", page_icon="🚀")
st.title("Object Detection with Custom YOLO Model")

# Sidebar for Model Selection
model_path = 'temp/best.pt' # Path in your GitHub repo
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

# Load Model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)

# File Uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to opencv image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Inference
    results = model.predict(image, conf=confidence)
    res_plotted = results[0].plot()
    
    # Display Result
    st.image(res_plotted, caption="Detected Image", use_column_width=True)
