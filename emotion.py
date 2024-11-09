import cv2
import numpy as np

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
    mouth_open = features['mouth_open']
    mouth_corners = features['mouth_corners']
    eyebrow_constriction = features['eyebrow_constriction']
    wrinkles = features['wrinkles']

    # Simple rule-based emotion classification
    if mouth_open and not wrinkles and eyebrow_constriction < 0.5:
        return "Happy"
    elif mouth_open and wrinkles and eyebrow_constriction > 0.5:
        return "Surprise"
    elif not mouth_open and wrinkles and eyebrow_constriction < 0.5:
        return "Angry"
    elif not mouth_open and not wrinkles and eyebrow_constriction > 0.5:
        return "Neutral"
    else:
        return "Disgust"


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

        # Feature extraction placeholders
        features = {
            'mouth_open': len(np.where(mouth_region > 0)[0]) > 500,  
            'mouth_corners': np.sum(mouth_region) / 255,  # Summing for intensity
            'eyebrow_constriction': 0.4,  # Placeholder value
            'wrinkles': np.sum(wrinkles) / 255 > 50  # Example threshold for wrinkles
        }

        # Classify emotion based on extracted features
        emotion = classify_emotion(features)
        print(f"Detected Emotion: {emotion}")

        # Draw face and emotion text on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image


def main():
    cap = cv2.VideoCapture(1)

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
main()
