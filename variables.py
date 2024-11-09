import cv2
import dlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Load dlib’s shape predictor for facial landmarks (ensure you have "shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Custom Feature Extractor Class
class EmotionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, mouth_open_threshold=500, wrinkle_intensity_threshold=50, eyebrow_constriction_threshold=0.5):
        self.mouth_open_threshold = mouth_open_threshold
        self.wrinkle_intensity_threshold = wrinkle_intensity_threshold
        self.eyebrow_constriction_threshold = eyebrow_constriction_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for img in X:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) == 0:
                # If no face is detected, return a default feature vector (optional: you can skip these cases)
                features.append([0, 0, 0])
                continue

            # Process the first detected face
            face = faces[0]
            landmarks = predictor(gray, face)

            # Mouth Opening Feature
            mouth_opening = np.linalg.norm(np.array([landmarks.part(51).x, landmarks.part(51).y]) -
                                           np.array([landmarks.part(59).x, landmarks.part(59).y]))
            mouth_open = mouth_opening > self.mouth_open_threshold

            # Eyebrow Constriction Feature
            eyebrow_distance = np.linalg.norm(np.array([landmarks.part(21).x, landmarks.part(21).y]) -
                                              np.array([landmarks.part(22).x, landmarks.part(22).y]))
            eyebrow_constriction = eyebrow_distance < self.eyebrow_constriction_threshold

            # Forehead Wrinkles Feature
            forehead_region = gray[face.top():face.top() + (face.bottom() - face.top()) // 4, face.left():face.right()]
            wrinkles = cv2.Canny(forehead_region, 100, 200)
            wrinkle_intensity = np.sum(wrinkles) / 255
            wrinkles_present = wrinkle_intensity > self.wrinkle_intensity_threshold

            # Append features (0 or 1 based on threshold evaluations)
            features.append([int(mouth_open), int(eyebrow_constriction), int(wrinkles_present)])

        return np.array(features)


# Load and preprocess the FER-2013 dataset
def load_fer2013_data(filepath='fer2013.csv'):
    data = pd.read_csv(filepath)

    # Separate features and labels
    X, y = [], []

    for index, row in data.iterrows():
        pixels = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)  # Reshape to 48x48
        # Convert to 3-channel BGR image for compatibility with dlib (from grayscale to BGR)
        img = cv2.cvtColor(pixels.astype('uint8'), cv2.COLOR_GRAY2BGR)
        X.append(img)
        y.append(row['emotion'])

    return X, y


# Load the dataset
X, y = load_fer2013_data()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with the custom feature extractor and classifier
pipeline = Pipeline([
    ('feature_extractor', EmotionFeatureExtractor()),  # Feature extraction step
    ('classifier', SVC())  # Classifier step
])

# Define the parameter grid for grid search
param_grid = {
    'feature_extractor__mouth_open_threshold': [300, 500, 700],
    'feature_extractor__wrinkle_intensity_threshold': [20, 50, 80],
    'feature_extractor__eyebrow_constriction_threshold': [0.3, 0.5, 0.7],
    'classifier__C': [0.1, 1, 10],  # SVM regularization parameter
    'classifier__gamma': [0.001, 0.01, 0.1]  # Kernel coefficient for SVM
}
print("runninf")

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Test with the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy with Best Parameters:", test_accuracy)
