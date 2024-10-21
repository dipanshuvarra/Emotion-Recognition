import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

def get_emotion(face_roi, eyes, smile):
    if len(eyes) >= 2:
        eye_height = eyes[0][3] 
        face_height = face_roi.shape[0]
        eye_ratio = eye_height / face_height
        
        if eye_ratio < 0.37: 
            return "Shocked"

    return "Joyful" if len(smile) > 0 else "Neutral"

def detect_mouth_openness(face_roi):
    mouth_roi = face_roi[face_roi.shape[0]//2:, :] 
    gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_mouth, 70, 255, cv2.THRESH_BINARY_INV)
    
    open_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    mouth_open_ratio = open_pixels / total_pixels
    
    return mouth_open_ratio > 0.4

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        smile = smile_cascade.detectMultiScale(face_roi, 1.55, 16)

        emotion = get_emotion(face_roi, eyes, smile)
        
        if detect_mouth_openness(face_roi):
            emotion = "Shocked"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
