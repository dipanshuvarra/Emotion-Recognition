import os
import cv2
import numpy as np
import dlib

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Landmark indices for mouth, eyes, and eyebrows
MOUTH_START, MOUTH_END = 48, 68
LEFT_EYE_START, LEFT_EYE_END = 36, 42
RIGHT_EYE_START, RIGHT_EYE_END = 42, 48
LEFT_EYEBROW_START, LEFT_EYEBROW_END = 17, 22
RIGHT_EYEBROW_START, RIGHT_EYEBROW_END = 22, 27

# Function to calculate Euclidean distance between two points
def euclidean_dist(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to extract features like mouth openness, eye openness, and eyebrow distance
def extract_features(landmarks):
    mouth_openness = euclidean_dist(landmarks[51], landmarks[57]) / euclidean_dist(landmarks[48], landmarks[54])
    left_eye_openness = (euclidean_dist(landmarks[37], landmarks[41]) + euclidean_dist(landmarks[38],
                                                                                       landmarks[40])) / 2
    right_eye_openness = (euclidean_dist(landmarks[43], landmarks[47]) + euclidean_dist(landmarks[44],
                                                                                        landmarks[46])) / 2
    eyebrow_distance = euclidean_dist(landmarks[21], landmarks[22])
    return mouth_openness, left_eye_openness, right_eye_openness, eyebrow_distance

# Function to extract facial landmarks from an image
def extract_landmarks(image):
    # Ensure image is in the correct format (uint8, 3 channels)
    if image is None or len(image.shape) != 3:  # 3 channels for color image (BGR)
        print("Error: Invalid image")
        return None

    # Resize image to improve face detection
    image = cv2.resize(image, (150, 150))

    # Convert the image to grayscale (uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)
    print(f"Detected faces: {len(faces)}")  # Debugging: Print number of faces detected

    # If no faces are detected, return None
    if len(faces) == 0:
        print("No faces detected.")
        return None

    # Process the first detected face
    landmarks = predictor(gray, faces[0])
    landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

    # Debugging: Draw landmarks on the image
    for (x, y) in landmarks_points:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw landmarks in green
    cv2.imshow("Landmarks", image)
    cv2.waitKey(10)  # Wait for 1 second (1000 milliseconds)
    cv2.destroyAllWindows()

    return landmarks_points

# Define the folder containing the images
main_folder = r"D:\Downloads\Facial-Emotion-Recognition-using-CNN-master\FER\test"  # Adjust path as needed
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize variables for emotion features
emotion_features = {emotion: [] for emotion in emotion_labels}

# Iterate through each emotion folder inside the main folder
for emotion in emotion_labels:
    emotion_folder = os.path.join(main_folder, emotion)  # Path to the emotion-specific subfolder

    if not os.path.exists(emotion_folder):
        print(f"Folder {emotion} not found. Skipping...")
        continue

    # List all images in the emotion folder
    image_files = [f for f in os.listdir(emotion_folder) if f.endswith('.jpg') or f.endswith('.png')]  # Adjust extensions as needed

    # Extract features for each image in the folder
    for i, image_file in enumerate(image_files):
        # Load the image
        image_path = os.path.join(emotion_folder, image_file)
        image = cv2.imread(image_path)

        # Check the image's shape and data type
        print(f"Processing image {i} from {emotion} folder, shape: {image.shape}, dtype: {image.dtype}")

        landmarks_points = extract_landmarks(image)


        if landmarks_points:
            print(f"Landmarks extracted for image {i} from {emotion} folder")
            features = extract_features(landmarks_points)
            emotion_features[emotion].append(features)
        else:
            print(f"No landmarks found for image {i} in {emotion} folder")


# Calculate mean and standard deviation for each emotion
emotion_stats = {}
for emotion, feature_list in emotion_features.items():
    if len(feature_list) > 0:  # Ensure we have features for that emotion
        feature_list = np.array(feature_list)
        mean = np.mean(feature_list, axis=0)
        std = np.std(feature_list, axis=0)
        emotion_stats[emotion] = {'mean': mean, 'std': std}

# Print out the mean and standard deviation for each emotion (optional)
k = 1

# Initialize emotion thresholds based on calculated mean and std
emotion_thresholds = {}

# Iterate over each emotion and set thresholds based on the mean and std
for emotion, stats in emotion_stats.items():
    mean = stats['mean']
    std = stats['std']

    # Dynamically calculate the thresholds for mouth_openness, eye_openness, and eyebrow_distance
    emotion_thresholds[emotion] = {
        'mouth_openness': (mean[0] - k * std[0], mean[0] + k * std[0]),
        'eye_openness': (mean[1] - k * std[1], mean[1] + k * std[1]),
        'eyebrow_distance': (mean[2] - k * std[2], mean[2] + k * std[2])
    }

# Print the updated thresholds for each emotion
for emotion, thresholds in emotion_thresholds.items():
    print(f"Emotion: {emotion}")
    print(f"Thresholds: {thresholds}")
