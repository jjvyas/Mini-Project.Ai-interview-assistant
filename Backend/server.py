# server.py
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from fer import FER
import os

app = FastAPI()

# Create a folder for uploaded videos and frames
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Open video with OpenCV
    cap = cv2.VideoCapture(video_path)
    detector = FER(mtcnn=True)  # FER face/emotion detector

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (FER expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze emotions
        emotions = detector.detect_emotions(rgb_frame)
        if emotions:
            results.append({
                "frame": frame_count,
                "emotions": emotions
            })
        
        frame_count += 1

    cap.release()

    return {"video": file.filename, "total_frames": frame_count, "emotion_results": results}