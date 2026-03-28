from fer import FER
import cv2

detector = FER()

def analyze_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return {"error": "Image not found"}

    result = detector.detect_emotions(img)

    return result