import cv2
import numpy as np

# Load the Haar Cascades for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return img_copy, faces  # Return both the image and the detected faces

def get_emotion(face_roi, smile):
    # Check for a smile to determine joy
    if len(smile) > 0:
        return "Joyful"
    return "Neutral"

def detect_mouth_openness(face_roi):
    mouth_roi = face_roi[face_roi.shape[0]//2:, :]
    gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_mouth, 70, 255, cv2.THRESH_BINARY_INV)
    
    open_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    mouth_open_ratio = open_pixels / total_pixels
    
    return mouth_open_ratio > 0.2 # Threshold for mouth openness

def detect_shock(face_roi):
    # You can refine this logic as per your requirements
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eye_roi = gray_face[0:int(face_roi.shape[0]/3), :]  # Assuming eyes are in the upper part
    _, thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)
    
    open_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    eye_open_ratio = open_pixels / total_pixels
    
    return eye_open_ratio < 0.2 # Adjust this threshold for shocked detection

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detected_faces_img, faces = detect_faces(face_cascade, frame)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(face_roi, 1.55, 16)

        # Determine the emotion based on the smile detection
        emotion = get_emotion(face_roi, smile)

        # Check for mouth openness to determine shock
        if detect_mouth_openness(face_roi):
            emotion = "Shocked"

        # Check for shocked based on eye characteristics
        if detect_shock(face_roi):
            emotion = "Shocked"

        # Display the emotion on the frame
        cv2.putText(detected_faces_img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', detected_faces_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
