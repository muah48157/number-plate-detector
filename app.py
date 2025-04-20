import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile

# Set page config
st.set_page_config(
    page_title="Number Plate Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

# Load model
@st.cache_resource
def load_model():
    return YOLO('models/number_plate_model.pt')

model = load_model()

# App header
st.title("🚗 Number Plate Detection System")
st.markdown("""
    Upload an image or video to detect license plates using our AI model.
    The system will automatically identify and highlight all license plates in your media.
""")

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=200)
    st.markdown("## Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.01)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses YOLOv8 to detect license plates in images and videos.
    The model was trained on a custom dataset of vehicle images.
    """)

# Main content
tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

with tab1:
    st.subheader("Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Read and display original image
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image
        with st.spinner("Detecting license plates..."):
            # Convert to OpenCV format
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run detection
            results = model.predict(img_array, conf=confidence)
            
            # Plot results
            res_plotted = results[0].plot()
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Display results
            with col2:
                st.image(res_plotted, caption="Detected License Plates", use_column_width=True)
            
            # Show detection info
            st.success("Detection Complete!")
            with st.expander("Detection Details"):
                for i, box in enumerate(results[0].boxes):
                    st.write(f"**License Plate {i+1}**")
                    st.write(f"- Confidence: {box.conf.item():.2f}")
                    st.write(f"- Bounding Box: {box.xyxy.tolist()[0]}")

with tab2:
    st.subheader("Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        
        st_frame = st.empty()
        st_progress = st.progress(0)
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.write(f"Video Info: {width}x{height} at {fps:.2f} FPS")
        
        # Prepare output
        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = model.predict(frame, conf=confidence)
            frame = results[0].plot()
            
            # Write frame
            out.write(frame)
            
            # Display every 5th frame
            if frame_count % 5 == 0:
                st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                             caption="Processed Frame", use_column_width=True)
            
            # Update progress
            frame_count += 1
            st_progress.progress(frame_count / total_frames)
        
        # Release resources
        cap.release()
        out.release()
        
        # Show download button
        st.success("Video processing complete!")
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="detected_plates.mp4",
                mime="video/mp4"
            )