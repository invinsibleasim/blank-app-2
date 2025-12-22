#import streamlit as st
#from ultralytics import YOLO
#import cv2
#import numpy as np
#from PIL import Image

# Set Page Config
#st.set_page_config(page_title="Custom YOLOv11 Detection", page_icon="🚀")
#st.title("Object Detection with Custom YOLO Model")

# Sidebar for Model Selection
#model_path = 'temp/best.pt' # Path in your GitHub repo
#confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

# Load Model
#@st.cache_resource
#def load_model(path):
    #return YOLO(path)

#model = load_model(model_path)

# File Uploader
#uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

#if uploaded_file is not None:
    # Convert file to opencv image
    #image = Image.open(uploaded_file)
    #image = np.array(image)
    
    # Inference
    #results = model.predict(image, conf=confidence)
    #res_plotted = results[0].plot()
    
    # Display Result
    #st.image(res_plotted, caption="Detected Image", use_column_width=True)


# app_yolo11_streamlit.py
# Streamlit web app: YOLO11 object detection on images and videos with confidence, class labels,
# downloadable annotated outputs.
# Author: Asim & M365 Copilot

import io
import os
from pathlib import Path
from typing import List, Dict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Ultralytics YOLO11
from ultralytics import YOLO

# ------------------------------
# Utility functions
# ------------------------------

# ---- Replace the sidebar selection with a fixed path to best.pt ----
from pathlib import Path

# Point this to your file; absolute paths are fine too
weights_path = "temp/best.pt"

with st.spinner("Loading YOLO11 model…"):
    model = load_model(weights_path)


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    """Load YOLO model and cache it for reuse."""
    model = YOLO(weights_path)
    return model


def draw_boxes(frame: np.ndarray, boxes, names: Dict[int, str], conf_thresh: float = 0.25) -> np.ndarray:
    """Draw bounding boxes and labels on a frame using YOLO results."""
    img = frame.copy()
    if boxes is None:
        return img

    # Generate a color palette for classes
    def get_color(idx: int):
        np.random.seed(idx)
        color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
        return color

    for i in range(len(boxes)):
        b = boxes[i]
        conf = float(b.conf[0]) if hasattr(b, 'conf') else float(b.conf)
        cls_idx = int(b.cls[0]) if hasattr(b, 'cls') else int(b.cls)
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0]) if hasattr(b, 'xyxy') else map(int, b.xyxy)
        color = get_color(cls_idx)
        label = f"{names.get(cls_idx, str(cls_idx))} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def results_to_df(res, names: Dict[int, str]) -> pd.DataFrame:
    """Convert YOLO prediction results to a tidy DataFrame."""
    rows = []
    boxes = res.boxes
    if boxes is None:
        return pd.DataFrame(columns=["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    for i in range(len(boxes)):
        b = boxes[i]
        cls_idx = int(b.cls[0]) if hasattr(b, 'cls') else int(b.cls)
        conf = float(b.conf[0]) if hasattr(b, 'conf') else float(b.conf)
        x1, y1, x2, y2 = map(float, b.xyxy[0]) if hasattr(b, 'xyxy') else map(float, b.xyxy)
        rows.append({
            "class_id": cls_idx,
            "class_name": names.get(cls_idx, str(cls_idx)),
            "confidence": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    df = pd.DataFrame(rows)
    return df


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="YOLO11 Object Detection", page_icon="🟡", layout="wide")
st.title("🟡 YOLO11 Object Detection (Images & Videos)")

st.sidebar.header("⚙️ Configuration")
# Model selection
model_choice = st.sidebar.selectbox(
    "Choose YOLO11 model weights",
    (
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
        "Upload custom .pt"
    ),
    index=0,
)
custom_model = None
if model_choice == "Upload custom .pt":
    custom_model = st.sidebar.file_uploader("Upload custom YOLO .pt weights", type=["pt"], accept_multiple_files=False)
    if custom_model is not None:
        # Save to a temporary path
        temp_weights_path = Path(st.session_state.get("_temp_weights", "temp_model.pt"))
        temp_weights_path.write_bytes(custom_model.read())
        weights_path = str(temp_weights_path)
    else:
        st.info("Upload a .pt file to proceed.")
        st.stop()
else:
    weights_path = model_choice

# Confidence & IoU thresholds
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
ious_thresh = st.sidebar.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.01)

# Load model
with st.spinner("Loading YOLO11 model…"):
    model = load_model(weights_path)

names = model.names if hasattr(model, 'names') else {i: str(i) for i in range(100)}

# Class filter
all_classes = list(names.values())
selected_classes = st.sidebar.multiselect("Filter classes (optional)", options=all_classes, default=[])
selected_class_ids = [k for k, v in names.items() if v in selected_classes] if selected_classes else None

# Processing options
max_det = st.sidebar.number_input("Max detections per image/frame", min_value=10, max_value=1000, value=300, step=10)
imgsz = st.sidebar.number_input("Image size (inference)", min_value=320, max_value=1280, value=640, step=64)

st.sidebar.markdown("---")
st.sidebar.caption("Model and framework powered by Ultralytics YOLO11.")

# Tabs for image and video
tab_img, tab_vid = st.tabs(["🖼️ Image", "🎥 Video"])

# ------------------------------
# Image tab
# ------------------------------
with tab_img:
    st.subheader("Image Inference")
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    cam_img = st.camera_input("Or capture from webcam (optional)")

    source_image = None
    if img_file is not None:
        source_image = Image.open(img_file).convert("RGB")
    elif cam_img is not None:
        source_image = Image.open(cam_img).convert("RGB")

    if source_image is not None:
        st.image(source_image, caption="Input image", use_column_width=True)
        # Run inference
        with st.spinner("Running detection…"):
            res_list = model.predict(
                source=np.array(source_image),
                conf=conf_thresh,
                iou=ious_thresh,
                classes=selected_class_ids,
                max_det=int(max_det),
                imgsz=int(imgsz),
                verbose=False,
            )
            res = res_list[0]
            df = results_to_df(res, names)

        # Annotate
        annotated = draw_boxes(np.array(source_image), res.boxes, names, conf_thresh)
        st.image(annotated, caption="Annotated image", use_column_width=True)

        # Show table & stats
        st.write("### Detections")
        st.dataframe(df)
        if not df.empty:
            counts = df["class_name"].value_counts().rename_axis("class").reset_index(name="count")
            st.write("### Class counts")
            st.dataframe(counts)

        # Download annotated image
        annotated_pil = Image.fromarray(annotated)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download annotated image (PNG)",
            data=buf.getvalue(),
            file_name="annotated.png",
            mime="image/png",
        )

# ------------------------------
# Video tab
# ------------------------------
with tab_vid:
    st.subheader("Video Inference")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    frame_skip = st.number_input("Frame skip (process every Nth frame)", min_value=1, max_value=20, value=1)

    if video_file is not None:
        # Save uploaded video to a temp path
        temp_video_path = Path("temp_input_video")
        temp_video_path.write_bytes(video_file.read())

        # Read video
        cap = cv2.VideoCapture(str(temp_video_path))
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = Path("annotated_output.mp4")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0.0, text="Processing video…")

            processed = 0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % int(frame_skip) != 0:
                    continue
                # Run inference per frame
                res_list = model.predict(
                    source=frame,
                    conf=conf_thresh,
                    iou=ious_thresh,
                    classes=selected_class_ids,
                    max_det=int(max_det),
                    imgsz=int(imgsz),
                    verbose=False,
                )
                res = res_list[0]
                annotated_frame = draw_boxes(frame, res.boxes, names, conf_thresh)
                writer.write(annotated_frame)
                processed += 1
                if total_frames:
                    progress.progress(min((frame_idx / total_frames), 1.0), text=f"Processing frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            st.success("Video processing complete.")
            # Show preview of first frame from the output video
            st.video(str(out_path))
            # Provide download.
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download annotated video (MP4)",
                    data=f.read(),
                    file_name="annotated_output.mp4",
                    mime="video/mp4",
                )

# ------------------------------
# Footer
# ------------------------------
st.markdown(
    """
    ---
    **Notes**:
    - Confidence and class filters control which detections are rendered and listed.
    - For large videos, increase *Frame skip* or reduce *Image size* for faster processing.
    - If you upload custom weights, ensure they are compatible with Ultralytics YOLO11.
    - This app uses the Ultralytics `YOLO` Python API. See documentation for details.
    """
)
