import cv2
import mediapipe as mp
import numpy as np
from pyicub.helper import iCub


# Initialize BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils


icub = iCub()
#        self.IGazeControl = 


cap = cv2.VideoCapture(0)

yaw_offset = None  # Offset to correct yaw when the head faces forward


# Define a new keypoint as the midpoint eyes
def find_mid_keypoint(landmarks, index):

    x = np.mean([landmarks[index[0]].x, landmarks[index[1]].x])
    y = np.mean([landmarks[index[0]].y, landmarks[index[1]].y])
    z = np.mean([landmarks[index[0]].z, landmarks[index[1]].z])

    return x, y, z



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            i_eye = [2,5]
            x_m, y_m, z_m = eyes_mid = find_mid_keypoint(landmarks, i_eye)
            m_xd = int(x_m * image_width)
            m_yd = int(y_m * image_height)

            print(m_xd, m_yd)
            cv2.circle(frame, (m_xd, m_yd), 5, (0, 0, 255), 1)  # Red dot for the new keypoint


            print("x: " + str(x_m) + " y: " + str(y_m) + " z: " + str(z_m))



    cv2.imshow('iTrackPeople', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

