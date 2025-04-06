import cv2
import numpy as np
import mediapipe as mp
import yarp
import argparse

class iTrackPeople:
    def __init__(self, display=True):
        self.display = display

        # Init YARP
        yarp.Network.init()

        # YARP input port for receiving images (e.g. from /grabber)
        self.input_port = yarp.BufferedPortImageRgb()
        self.input_port.open("/iTrackPeople/image:i")

        # YARP output port for sending midpoint coordinates
        self.output_port = yarp.BufferedPortBottle()
        self.output_port.open("/iTrackPeople/eyes:o")

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    def find_eye_midpoint(self, landmarks, indices):
        x = np.mean([landmarks[i].x for i in indices])
        y = np.mean([landmarks[i].y for i in indices])
        z = np.mean([landmarks[i].z for i in indices])
        return x, y, z

    def run(self):
        print("[iTrackPeople] Running... Press Ctrl+C to stop.")
        while True:
            yarp_image = self.input_port.read()
            if yarp_image is None:
                continue

            # Convert YARP image to OpenCV format
            h, w = yarp_image.height(), yarp_image.width()
            img = np.zeros((h, w, 3), dtype=np.uint8)

            for y in range(h):
                for x in range(w):
                    pixel = yarp_image.pixel(x, y)
                    img[y, x, 0] = pixel.b
                    img[y, x, 1] = pixel.g
                    img[y, x, 2] = pixel.r
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



            # Process with MediaPipe
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if all(lm[i].visibility > 0.5 for i in [2, 5]):  # Left eye: 2, Right eye: 5
                    u, v, z = self.find_eye_midpoint(lm, [2, 5])

                    # Output coordinates via YARP
                    bottle = self.output_port.prepare()
                    bottle.clear()
                    bottle.addFloat32(u)
                    bottle.addFloat32(v)
                    bottle.addFloat32(z)
                    self.output_port.write()

                    # Visualization
                    if self.display:
                        cx, cy = int(u * w), int(v * h)
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

            if self.display:
                cv2.imshow("iTrackPeople", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cleanup()

    def cleanup(self):
        print("[iTrackPeople] Shutting down...")
        self.input_port.close()
        self.output_port.close()
        cv2.destroyAllWindows()
        yarp.Network.fini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true', help="Enable OpenCV window")
    args = parser.parse_args()

    tracker = iTrackPeople(display=args.display)
    tracker.run()
