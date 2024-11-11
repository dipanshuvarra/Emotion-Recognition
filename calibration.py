import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in the same directory

def get_mouth_openness(landmarks):
    # Mouth landmarks: points 48 to 59
    mouth_points = [landmarks.part(i) for i in range(48, 60)]  # Use part(i) to get each point
    # Calculate mouth openness (distance between top and bottom lips)
    mouth_height = np.linalg.norm([mouth_points[3].x - mouth_points[9].x, mouth_points[3].y - mouth_points[9].y])
    return mouth_height

def get_eye_openness(landmarks):
    left_eye = [landmarks.part(i) for i in range(36, 42)]
    right_eye = [landmarks.part(i) for i in range(42, 48)]

    left_eye_width = np.linalg.norm([left_eye[0].x - left_eye[3].x, left_eye[0].y - left_eye[3].y])
    left_eye_height = np.linalg.norm([left_eye[1].x - left_eye[5].x, left_eye[1].y - left_eye[5].y])
    right_eye_width = np.linalg.norm([right_eye[0].x - right_eye[3].x, right_eye[0].y - right_eye[3].y])
    right_eye_height = np.linalg.norm([right_eye[1].x - right_eye[5].x, right_eye[1].y - right_eye[5].y])

    left_eye_ratio = left_eye_width / left_eye_height
    right_eye_ratio = right_eye_width / right_eye_height

    return (left_eye_ratio + right_eye_ratio) / 2  # Average of both eyes

def get_eyebrow_eye_distance(landmarks):
    left_eyebrow = [landmarks.part(i) for i in range(17, 22)]
    right_eyebrow = [landmarks.part(i) for i in range(22, 27)]

    left_eye = [landmarks.part(i) for i in range(36, 42)]
    right_eye = [landmarks.part(i) for i in range(42, 48)]

    left_distances = []
    for eyebrow_point in left_eyebrow:
        for eye_point in left_eye:
            dist = np.linalg.norm([eyebrow_point.x - eye_point.x, eyebrow_point.y - eye_point.y])
            left_distances.append(dist)

    right_distances = []
    for eyebrow_point in right_eyebrow:
        for eye_point in right_eye:
            dist = np.linalg.norm([eyebrow_point.x - eye_point.x, eyebrow_point.y - eye_point.y])
            right_distances.append(dist)

    left_eyebrow_distance = min(left_distances)
    right_eyebrow_distance = min(right_distances)

    return (left_eyebrow_distance + right_eyebrow_distance) / 2

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        mouth_openness = get_mouth_openness(landmarks)
        eye_openness = get_eye_openness(landmarks)
        eyebrow_eye_distance = get_eyebrow_eye_distance(landmarks)

        print(f"Mouth Openness: {mouth_openness}")
        print(f"Eye Openness: {eye_openness}")
        print(f"Eyebrow to Eye Distance: {eyebrow_eye_distance}")

        cv2.putText(image, f"Mouth Openness: {mouth_openness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.putText(image, f"Eye Openness: {eye_openness:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Eyebrow to Eye Distance: {eyebrow_eye_distance:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        for i in range(48, 60):  # Mouth points
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
        for i in range(36, 42):  # Left eye points
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
        for i in range(42, 48):  # Right eye points
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
        for i in range(17, 22):  # Left eyebrow points
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
        for i in range(22, 27):  # Right eyebrow points
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)

    return image

def main():
    cap = cv2.VideoCapture(1)  # Use 0 for webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        result_image = process_image(frame)

        cv2.imshow("Emotion Features", result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
