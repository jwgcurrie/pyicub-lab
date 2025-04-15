import cv2
import numpy as np
import mediapipe as mp
import yarp
import sys

VOCAB_QUIT = yarp.createVocab32("q", "u", "i", "t")

class iTrackPeople(yarp.RFModule):
    def __init__(self):
        super().__init__()
        self.display = False
        self.period = 0.1  # default update period

        # Tracking & MediaPipe
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

        # YARP ports
        self.input_port = yarp.BufferedPortImageRgb()
        self.output_port = yarp.BufferedPortBottle()
        self.rpc_port = yarp.RpcClient()
        self.cmd_port = yarp.Port()  # for receiving commands

    def configure(self, rf):
        print("[iTrackPeople] Configuring module...")

        self.display = rf.check("display") and rf.find("display").asBool()
        self.period = rf.check("period") and rf.find("period").asFloat64() or 0.1

        self.input_port.open("/iTrackPeople/image:i")
        self.output_port.open("/iTrackPeople/eyes:o")
        self.rpc_port.open("/iTrackPeople/rpc:o")
        self.cmd_port.open("/iTrackPeople/cmd:rpc")

        self.attach(self.cmd_port)  # connect respond() to RPC input
        return True

    def getPeriod(self):
        return self.period

    def updateModule(self):
        yarp_image = self.input_port.read()
        if yarp_image is None:
            return True

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

        visibility_criteria = 0.5
        keypoints = [2, 5] # keypoints = [left_eye, right_eye]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if all(lm[i].visibility > visibility_criteria for i in keypoints):
                u, v, _ = self.find_eye_midpoint(lm, keypoints)
                cx, cy = int(u * w), int(v * h)

                if 0 <= cx < w and 0 <= cy < h:
                    # Send 2D midpoint
                    bottle = self.output_port.prepare()
                    bottle.clear()
                    bottle.addInt32(cx)
                    bottle.addInt32(cy)
                    self.output_port.write()

                    if self.display:
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    print("[iTrackPeople] Midpoint out of bounds.")
            else:
                print("[iTrackPeople] Eyes not visible.")
        else:
            print("[iTrackPeople] No landmarks detected.")

        if self.display:
            cv2.imshow("iTrackPeople", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        return True

    def respond(self, command, reply):
        print(f"[iTrackPeople] Received command: {command.toString()}")

        if command.check("period"):
            self.period = command.find("period").asFloat64()
            reply.addString("ack")
            return True
        elif command.get(0).asVocab() == VOCAB_QUIT:
            reply.addString("bye")
            return False
        else:
            reply.addString("nack")
            reply.addString("unknown command")
            return True

    def find_eye_midpoint(self, landmarks, indices):
        x = np.mean([landmarks[i].x for i in indices])
        y = np.mean([landmarks[i].y for i in indices])
        z = np.mean([landmarks[i].z for i in indices])
        return x, y, z

    def interruptModule(self):
        print("[iTrackPeople] Interrupting...")
        return True

    def close(self):
        print("[iTrackPeople] Closing ports and cleaning up...")
        self.input_port.close()
        self.output_port.close()
        self.rpc_port.close()
        self.cmd_port.close()
        if self.display:
            cv2.destroyAllWindows()
        return True


if __name__ == "__main__":
    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("iTrackPeople")
    rf.setDefaultConfigFile("config.ini")
    rf.configure(sys.argv)

    module = iTrackPeople()
    module.runModule(rf)
