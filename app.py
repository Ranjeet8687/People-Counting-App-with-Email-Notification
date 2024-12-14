import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import io

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_people(image):
    results = model(image)
    detections = results.pandas().xyxy[0]
    person_count = len(detections[detections['name'] == 'person'])
    rendered_image = np.squeeze(results.render())
    return rendered_image, person_count

def send_email(person_count, recipient_email, image=None):
    sender_email = "csaiml22140@glbitm.ac.in"
    sender_password = "spco awfd wwwn unnt"
    subject = "People Counting Report"
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    email_body = f"The number of people detected: {person_count}"
    message.attach(MIMEText(email_body, "plain"))

    if image is not None:
        with io.BytesIO() as buffer:
            img = Image.fromarray(image)
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            img_data = buffer.read()
        image_attachment = MIMEImage(img_data, name="detection_result.jpg")
        message.attach(image_attachment)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

st.title("Real-Time People Counting App with Email Notification")
st.subheader("Upload an image or video to count the number of people.")

uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file:
    recipient_email = st.text_input("Enter recipient email address:")
    
    if uploaded_file.name.endswith(('jpg', 'png', 'jpeg')):
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        rendered_image, person_count = detect_people(image_np)
        st.image(rendered_image, caption=f"Detected People: {person_count}", use_column_width=True)

        if st.button("Send Email"):
            if recipient_email:
                send_email(person_count, recipient_email, image=rendered_image)
            else:
                st.error("Please enter a valid email address.")
    
    elif uploaded_file.name.endswith('mp4'):
        st.write("Processing video... This may take some time.")
        st.error("Email sending for videos is not implemented yet.")
