import cv2
import numpy as np
from collections import Counter

# Load Haar cascades for face, eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variables to store data
focus_and_emotion_data = []
final_summary_data = []


# Function to detect concentration level based on eye positions
def detect_concentration(eyes, face_gray):
    eye_positions = []
    for (ex, ey, ew, eh) in eyes[:2]:  # Consider the first two eyes detected
        eye_region = face_gray[ey:ey + eh, ex:ex + ew]
        _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            pupil_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(pupil_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                eye_positions.append(cx / ew)  # Normalize x-position

    if len(eye_positions) == 2:
        if all(0.4 <= pos <= 0.6 for pos in eye_positions):
            return "High"
        elif any(0.3 <= pos <= 0.7 for pos in eye_positions):
            return "Medium"
        else:
            return "Low"
    elif len(eye_positions) == 1:
        return "Medium"
    return "Low"


# Function to detect emotion based on face features
def detect_emotion(face_img):
    # Detect mouth, eyebrows, and forehead wrinkles to classify emotions
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    cr_channel = ycrcb[:, :, 1]
    _, mouth_mask = cv2.threshold(cr_channel, 150, 255, cv2.THRESH_BINARY)
    mouth_region = mouth_mask[int(face_img.shape[0] * 0.5):]  # Lower half

    gray_forehead = cv2.cvtColor(face_img[:int(face_img.shape[0] * 0.25), :], cv2.COLOR_BGR2GRAY)
    wrinkles = cv2.Canny(gray_forehead, 100, 200)

    features = {
        'mouth_open': len(np.where(mouth_region > 0)[0]) > 500,
        'wrinkles': np.sum(wrinkles) / 255 > 50  # Threshold for wrinkles
    }

    if features['mouth_open'] and not features['wrinkles']:
        return "Happy"
    elif features['mouth_open'] and features['wrinkles']:
        return "Surprise"
    elif not features['mouth_open'] and features['wrinkles']:
        return "Angry"
    else:
        return "Neutral"


# Process each frame, capture concentration and emotion
cap = cv2.VideoCapture('ad.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    interval = frame_rate  # Analyzing every second
    second_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        concentration_level = "Low"
        emotion = "Neutral"

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(face_gray)
                concentration_level = detect_concentration(eyes, face_gray)
                emotion = detect_emotion(frame[y:y + h, x:x + w])

                # Draw rectangles and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{concentration_level}, {emotion}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Capture data every second
        if second_counter % interval == 0:
            focus_and_emotion_data.append((concentration_level, emotion))

        second_counter += 1
        cv2.imshow('Focus and Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Summarize most frequent labels in 5-second intervals
    for i in range(0, len(focus_and_emotion_data), 5):
        window_data = focus_and_emotion_data[i:i + 5]
        concentration_levels = [data[0] for data in window_data]
        emotions = [data[1] for data in window_data]
        most_common_concentration = Counter(concentration_levels).most_common(1)[0][0]
        most_common_emotion = Counter(emotions).most_common(1)[0][0]
        final_summary_data.append((most_common_concentration, most_common_emotion))

    cap.release()
    cv2.destroyAllWindows()

print("Focus and Emotion Data (every second):", focus_and_emotion_data)
print("Final Summary (5-second intervals):", final_summary_data)
