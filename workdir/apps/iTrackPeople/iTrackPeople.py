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

        # YARP output port for sending pixel midpoint
        self.output_port = yarp.BufferedPortBottle()
        self.output_port.open("/iTrackPeople/eyes:o")

        # Output port for 3D head position in iCub's reference frame
        self.output_3d_port = yarp.BufferedPortBottle()
        self.output_3d_port.open("/iTrackPeople/head3D:o")

        # RPC port to iKinGazeCtrl
        self.rpc_port = yarp.RpcClient()
        self.rpc_port.open("/iTrackPeople/rpc:o")
        yarp.Network.connect("/iTrackPeople/rpc:o", "/iKinGazeCtrl/rpc")

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    def find_eye_midpoint(self, landmarks, indices):
        x = np.mean([landmarks[i].x for i in indices])
        y = np.mean([landmarks[i].y for i in indices])
        z = np.mean([landmarks[i].z for i in indices])
        return x, y, z
    
    def get_3d_point(self, u, v, z=1.0, eye='left'):
        cmd = yarp.Bottle()
        response = yarp.Bottle()

        cmd.addString("get")
        cmd.addString("3D")
        cmd.addString("mono")

        inner = yarp.Bottle()
        inner.addString(eye)
        inner.addInt32(u)
        inner.addInt32(v)
        inner.addFloat32(z)
        cmd.addList().read(inner)

        print(f"[iTrackPeople] Sending RPC: get 3D mono ('{eye}' {u} {v} {z})")

        if self.rpc_port.write(cmd, response):
            print(f"[iTrackPeople] RPC response: {response.toString()}")
            if response.get(0).asString() == "ack":
                coords = response.get(1).asList()
                x = coords.get(0).asFloat64()
                y = coords.get(1).asFloat64()
                z = coords.get(2).asFloat64()
                return x, y, z
            else:
                print("[iTrackPeople] RPC call failed (nack).")
        else:
            print("[iTrackPeople] RPC communication error.")
        return None

    def run(self):
        print("[iTrackPeople] Running... Press Ctrl+C to stop.")
        while True:
            yarp_image = self.input_port.read()
            if yarp_image is None:
                continue

            h, w = yarp_image.height(), yarp_image.width()
            img = np.zeros((h, w, 3), dtype=np.uint8)

            for y in range(h):
                for x in range(w):
                    pixel = yarp_image.pixel(x, y)
                    img[y, x, 0] = pixel.b
                    img[y, x, 1] = pixel.g
                    img[y, x, 2] = pixel.r

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if all(lm[i].visibility > 0.5 for i in [2, 5]):
                    u, v, _ = self.find_eye_midpoint(lm, [2, 5])
                    cx, cy = int(u * w), int(v * h)

                    if 0 <= cx < w and 0 <= cy < h:
                        # Send 2D midpoint via YARP
                        bottle = self.output_port.prepare()
                        bottle.clear()
                        bottle.addInt32(cx)
                        bottle.addInt32(cy)
                        self.output_port.write()

                        # Get and send 3D head position
                        head_pos = self.get_3d_point(cx, cy, z=1.0)
                        if head_pos:
                            x3d, y3d, z3d = head_pos
                            print(f"[iTrackPeople] Head 3D position: x={x3d:.3f}, y={y3d:.3f}, z={z3d:.3f}")

                            bottle3D = self.output_3d_port.prepare()
                            bottle3D.clear()
                            bottle3D.addFloat32(x3d)
                            bottle3D.addFloat32(y3d)
                            bottle3D.addFloat32(z3d)
                            self.output_3d_port.write()

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
        self.output_3d_port.close()
        self.rpc_port.close()
        cv2.destroyAllWindows()
        yarp.Network.fini()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true', help="Enable OpenCV window")
    args = parser.parse_args()

    tracker = iTrackPeople(display=args.display)
    tracker.run()
