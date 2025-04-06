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

            # Convert YARP image to OpenCV (BGR order for OpenCV)
            h, w = yarp_image.height(), yarp_image.width()
            img = np.zeros((h, w, 3), dtype=np.uint8)

            for y in range(h):
                for x in range(w):
                    pixel = yarp_image.pixel(x, y)
                    img[y, x, 0] = pixel.b
                    img[y, x, 1] = pixel.g
                    img[y, x, 2] = pixel.r

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if all(lm[i].visibility > 0.5 for i in [2, 5]):
                    u, v, z = self.find_eye_midpoint(lm, [2, 5])

                    # Convert to pixel coordinates
                    cx, cy = int(u * w), int(v * h)

                    if 0 <= cx < w and 0 <= cy < h:
                        # Send pixel midpoint via YARP
                        bottle = self.output_port.prepare()
                        bottle.clear()
                        bottle.addInt32(cx)
                        bottle.addInt32(cy)
                        bottle.addFloat32(z)  # Still normalized z
                        self.output_port.write()

                        if self.display:
                            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    else:
                        print("[iTrackPeople] Pixel midpoint out of bounds: cx=%d, cy=%d" % (cx, cy))
                else:
                    print("[iTrackPeople] Landmarks not confident enough.")
            else:
                print("[iTrackPeople] No pose landmarks detected.")

            if self.display:
                cv2.imshow("iTrackPeople", img)
                if cv2.waitKey(5) & 0xFF == ord('q'):
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
