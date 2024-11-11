import cv2
import numpy as np

# Define the emotion thresholds (update these dynamically as per your previous calculations)
emotion_thresholds = {
    'angry': {'mouth_openness': (0.23, 0.66), 'eye_openness': (4.42, 8.53), 'eyebrow_distance': (4.44, 8.57)},
    'disgust': {'mouth_openness': (0.23, 0.48), 'eye_openness': (3.76, 7.88), 'eyebrow_distance': (3.79, 7.99)},
    'fear': {'mouth_openness': (0.22, 0.61), 'eye_openness': (4.81, 9.78), 'eyebrow_distance': (4.86, 9.83)},
    'happy': {'mouth_openness': (0.25, 0.52), 'eye_openness': (4.14, 7.90), 'eyebrow_distance': (4.14, 7.92)},
    'sad': {'mouth_openness': (0.20, 0.50), 'eye_openness': (4.30, 8.62), 'eyebrow_distance': (4.25, 8.60)},
    'surprise': {'mouth_openness': (0.38, 0.85), 'eye_openness': (7.09, 11.80), 'eyebrow_distance': (7.17, 11.76)},
    'neutral': {'mouth_openness': (0.22, 0.47), 'eye_openness': (4.98, 8.66), 'eyebrow_distance': (5.01, 8.72)}
}

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_face(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def detect_eyes(face_img):
    eyes = eye_cascade.detectMultiScale(face_img, 1.1, 4)
    return eyes


def detect_mouth_and_eyebrows(face_img):
    # Convert image to YCrCb color space for Cr channel analysis
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    cr_channel = ycrcb[:, :, 1]

    # Threshold Cr channel to detect lips
    _, mouth_mask = cv2.threshold(cr_channel, 150, 255, cv2.THRESH_BINARY)

    # Use edge detection and morphology to highlight mouth corners
    edges = cv2.Canny(mouth_mask, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Apply a region mask to locate the mouth area more accurately
    mouth_region = dilated[int(face_img.shape[0] * 0.5):]  # lower half of face
    return mouth_region


def detect_forehead_wrinkles(face_img):
    # Define forehead region
    forehead_region = face_img[:int(face_img.shape[0] * 0.25), :]  # top 25% of face

    # Convert to grayscale and apply Canny edge detection for wrinkles
    gray_forehead = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
    wrinkles = cv2.Canny(gray_forehead, 100, 200)
    return wrinkles


def classify_emotion(features):
    mouth_openness = features['mouth_openness']
    eye_openness = features['eye_openness']
    eyebrow_distance = features['eyebrow_distance']

    # Print extracted features for debugging
    print(f"Extracted Features: Mouth Openness: {mouth_openness}, Eye Openness: {eye_openness}, Eyebrow Distance: {eyebrow_distance}")

    # Compare the features with the emotion thresholds
    for emotion, thresholds in emotion_thresholds.items():
        mouth_openness_range = thresholds['mouth_openness']
        eye_openness_range = thresholds['eye_openness']
        eyebrow_distance_range = thresholds['eyebrow_distance']

        # Print the thresholds for debugging
        print(f"Comparing to Emotion: {emotion} - Mouth Openness: {mouth_openness_range}, Eye Openness: {eye_openness_range}, Eyebrow Distance: {eyebrow_distance_range}")

        # Check if features are within the thresholds for the current emotion
        if (mouth_openness_range[0] <= mouth_openness <= mouth_openness_range[1] and
            eye_openness_range[0] <= eye_openness <= eye_openness_range[1] and
            eyebrow_distance_range[0] <= eyebrow_distance <= eyebrow_distance_range[1]):
            return emotion

    return "Unknown"  # Default if no emotion matched


def process_image(image):
    # Detect face
    faces = detect_face(image)
    for (x, y, w, h) in faces:
        face_img = image[y:y + h, x:x + w]

        # Detect eyes
        eyes = detect_eyes(face_img)

        # Calculate mouth opening and eyebrow constriction
        mouth_region = detect_mouth_and_eyebrows(face_img)
        wrinkles = detect_forehead_wrinkles(face_img)

        # Feature extraction
        mouth_openness = len(np.where(mouth_region > 0)[0]) / float(mouth_region.size)  # Normalized mouth openness
        eye_openness = len(eyes)  # Number of detected eyes
        eyebrow_distance = np.mean([np.linalg.norm(eyes[0][:2] - eyes[1][:2]) if len(eyes) > 1 else 0])  # Placeholder

        features = {
            'mouth_openness': mouth_openness,
            'eye_openness': eye_openness,
            'eyebrow_distance': eyebrow_distance,
        }

        # Debugging: Print the extracted features before classification
        print(f"Processed Features: Mouth Openness: {mouth_openness}, Eye Openness: {eye_openness}, Eyebrow Distance: {eyebrow_distance}")

        # Classify emotion based on extracted features and thresholds
        emotion = classify_emotion(features)
        print(f"Detected Emotion: {emotion}")

        # Draw face and emotion text on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image



def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break
        result_image = process_image(frame)
        cv2.imshow('Emotion Detection', result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
