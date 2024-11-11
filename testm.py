import cv2
from deepface import DeepFace
from collections import Counter

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Counters for emotion and concentration level
emotion_counts = Counter()
concentration_counts = Counter()

def analyze_emotion_and_concentration(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No Face Detected", "Low"

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Handle single or multiple faces in the result
        if isinstance(result, list):
            result = result[0]

        # Get dominant emotion and update counters
        emotion = result['dominant_emotion']
        emotion_counts[emotion] += 1

        # Placeholder for concentration logic
        concentration_level = "High" if emotion in ["neutral", "happy"] else "Low"
        concentration_counts[concentration_level] += 1

        # Draw emotion and concentration level on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{emotion}, {concentration_level}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return emotion, concentration_level

def main():
    # Load the advertisement video and open the webcam
    ad_cap = cv2.VideoCapture('ad.mp4')
    webcam_cap = cv2.VideoCapture(0)

    while ad_cap.isOpened() and webcam_cap.isOpened():
        # Read frames from both ad video and webcam
        ret_ad, ad_frame = ad_cap.read()
        ret_webcam, webcam_frame = webcam_cap.read()

        if not ret_ad or not ret_webcam:
            break

        # Process webcam frame for emotion analysis
        emotion, concentration = analyze_emotion_and_concentration(webcam_frame)

        # Show both frames in separate windows
        cv2.imshow('Advertisement', ad_frame)
        cv2.imshow('Emotion & Concentration Detection', webcam_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    ad_cap.release()
    webcam_cap.release()
    cv2.destroyAllWindows()

    # Print the most common emotion and concentration level
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    most_common_concentration = concentration_counts.most_common(1)[0][0]
    print(f"Most Common Emotion: {most_common_emotion}")
    print(f"Most Common Concentration Level: {most_common_concentration}")

if __name__ == "__main__":
    main()
