import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Counters and thresholds
frame_count = 0
high_concentration_frames = 0
medium_concentration_frames = 0
low_concentration_frames = 0
max_medium_frames = 10  # Example threshold for medium concentration
max_low_frames = 5  # Example threshold for low concentration


def detect_pupil_position(eye_region):
    # Convert eye region to grayscale and apply a threshold to detect the darkest area (pupil)
    _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour, assuming it's the pupil
        pupil_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(pupil_contour)

        if M['m00'] != 0:
            # Calculate pupil center
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)

    return None


# Start video capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    concentration_level = "High"  # Default level

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 2:
                eye_positions = []

                # For each detected eye, determine if the pupil is centered
                for (ex, ey, ew, eh) in eyes[:2]:  # Consider only the first two detected eyes
                    eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                    pupil_position = detect_pupil_position(eye_roi_gray)

                    if pupil_position:
                        eye_positions.append(pupil_position[0] / ew)  # Normalize pupil x-position in eye width
                        cv2.circle(roi_color[ey:ey + eh, ex:ex + ew], pupil_position, 3, (0, 255, 0), -1)

                # Determine concentration level based on pupil positions
                if all(0.4 <= pos <= 0.6 for pos in eye_positions):
                    # Both pupils are centered; high concentration
                    high_concentration_frames += 1
                elif any(0.3 <= pos <= 0.7 for pos in eye_positions):
                    # One or both pupils are slightly off-center; medium concentration
                    medium_concentration_frames += 1
                    concentration_level = "Medium"
                else:
                    # Both pupils are significantly off-center; low concentration
                    low_concentration_frames += 1
                    concentration_level = "Low"

            elif len(eyes) == 1:
                # One eye detected, assume medium concentration
                medium_concentration_frames += 1
                concentration_level = "Medium"
            else:
                # No eyes detected, low concentration due to head rotation
                low_concentration_frames += 1
                concentration_level = "Low"

    else:
        # Face not detected, low concentration due to head turn
        low_concentration_frames += 1
        concentration_level = "Low"

    # Display concentration level on frame
    cv2.putText(frame, f"Concentration Level: {concentration_level}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Concentration Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Plot concentration data (replace with actual time or frame count)
plt.plot(range(frame_count), [high_concentration_frames, medium_concentration_frames, low_concentration_frames],
         label="Concentration")
plt.xlabel('Frames')
plt.ylabel('Concentration Levels')
plt.legend(["High", "Medium", "Low"])
plt.title('Student Concentration Over Time')
plt.show()
