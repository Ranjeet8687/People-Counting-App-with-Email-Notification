import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_people(image):
    # Perform detection
    results = model(image)
    detections = results.pandas().xyxy[0]  # Extract detections in a DataFrame
    person_count = len(detections[detections['name'] == 'person'])  # Filter 'person'
    # Render the image with detections
    rendered_image = np.squeeze(results.render())  # Get the processed image
    return rendered_image, person_count  # Return rendered image and count

# Streamlit UI
st.title("Real-Time People Counting App")
st.subheader("Upload an image or video to count the number of people.")

# Option to upload photo or video
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file:
    if uploaded_file.name.endswith(('jpg', 'png', 'jpeg')):
        # For photos
        image = Image.open(uploaded_file).convert('RGB')  # Open the image
        image_np = np.array(image)  # Convert to NumPy array
        rendered_image, person_count = detect_people(image_np)  # Detect people
        st.image(rendered_image, caption=f"Detected People: {person_count}", use_column_width=True)  # Display image
    elif uploaded_file.name.endswith('mp4'):
        # For videos (frame-by-frame processing)
        st.write("Processing video... This may take some time.")
        temp_file = "temp_video.mp4"  # Save uploaded video to a temp file
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        
        # Video processing
        cap = cv2.VideoCapture(temp_file)
        frame_count = 0
        person_count_total = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:  # Process every 10th frame
                results = model(frame)
                detections = results.pandas().xyxy[0]
                person_count_frame = len(detections[detections['name'] == 'person'])
                person_count_total += person_count_frame
                frame = np.squeeze(results.render())  # Render frame
            
        cap.release()
        st.write(f"Total Frames Processed: {frame_count}")
        st.write(f"Estimated Total People Count (in video): {person_count_total}")
