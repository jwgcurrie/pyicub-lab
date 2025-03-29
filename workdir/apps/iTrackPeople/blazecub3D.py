import cv2
import mediapipe as mp
import numpy as np
from pyicub.helper import iCub

# Initialize BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Initialize iCub
icub = iCub()
cap = cv2.VideoCapture(0)

# Camera parameters (approximate, should be calibrated)
FOCAL_LENGTH = 800  # Example focal length in pixels (change based on actual camera)
REAL_WORLD_SCALE = 0.3  # Scale factor to convert z-depth to meters (adjust as needed)

# Define a new keypoint as the midpoint between the eyes
def find_mid_keypoint(landmarks, index):
    x = np.mean([landmarks[index[0]].x, landmarks[index[1]].x])
    y = np.mean([landmarks[index[0]].y, landmarks[index[1]].y])
    z = np.mean([landmarks[index[0]].z, landmarks[index[1]].z])  # Depth estimation
    return x, y, z

# Convert normalized image coordinates to real-world 3D coordinates
def image_to_3d(x_norm, y_norm, z_norm, image_width, image_height):
    # Convert to pixel space
    x_pixel = x_norm * image_width
    y_pixel = y_norm * image_height

    # Estimate real-world depth (scaling the normalized depth)
    z_meters = z_norm * REAL_WORLD_SCALE  # Convert normalized depth to meters

    # Convert to real-world X, Y coordinates (assuming pinhole camera model)
    x_meters = (x_pixel - image_width / 2) * z_meters / FOCAL_LENGTH
    y_meters = (y_pixel - image_height / 2) * z_meters / FOCAL_LENGTH

    return x_meters, y_meters, z_meters

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.resize(rgb_frame, (640, 480))
    image_height, image_width, _ = frame.shape

    # Process frame with BlazePose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get head keypoint coordinates
        nose = landmarks[0]
        left_eye = landmarks[2]
        right_eye = landmarks[5]

        # Ensure key landmarks are visible
        if all(l.visibility > 0.5 for l in [nose, left_eye, right_eye]):
            i_eye = [2, 5]  # Indices for left and right eyes
            x_m, y_m, z_m = find_mid_keypoint(landmarks, i_eye)

            # Convert to real-world 3D coordinates
            X, Y, Z = image_to_3d(x_m, y_m, z_m, image_width, image_height)

            # Draw the midpoint in the 2D image
            m_xd = int(x_m * image_width)
            m_yd = int(y_m * image_height)
            cv2.circle(frame, (m_xd, m_yd), 5, (0, 0, 255), 1)  # Red dot for the new keypoint

            print(f"Image space (normalized): x={x_m:.3f}, y={y_m:.3f}, z={z_m:.3f}")
            print(f"Real-world 3D: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
            
            icub.gaze.lookAtFixationPoint(Z * 10, -X * 10, Y* 10, waitMotionDone=False)

    cv2.imshow('iTrackPeople', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
