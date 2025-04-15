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
        self.output_3d_port = yarp.BufferedPortBottle()
        self.rpc_port = yarp.RpcClient()
        self.cmd_port = yarp.Port()  # for receiving commands

    def configure(self, rf):
        print("[iTrackPeople] Configuring module...")

        self.display = rf.check("display") and rf.find("display").asBool()
        self.period = rf.check("period") and rf.find("period").asFloat64() or 0.1

        self.input_port.open("/iTrackPeople/image:i")
        self.output_port.open("/iTrackPeople/eyes:o")
        self.output_3d_port.open("/iTrackPeople/head3D:o")
        self.rpc_port.open("/iTrackPeople/rpc:o")
        self.cmd_port.open("/iTrackPeople/cmd:rpc")

        self.attach(self.cmd_port)  # connect respond() to RPC input

        yarp.Network.connect("/iTrackPeople/rpc:o", "/iKinGazeCtrl/rpc")

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

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if all(lm[i].visibility > 0.5 for i in [2, 5]):
                u, v, _ = self.find_eye_midpoint(lm, [2, 5])
                cx, cy = int(u * w), int(v * h)

                if 0 <= cx < w and 0 <= cy < h:
                    # Send 2D midpoint
                    bottle = self.output_port.prepare()
                    bottle.clear()
                    bottle.addInt32(cx)
                    bottle.addInt32(cy)
                    self.output_port.write()

                    # Send 3D position
                    head_pos = self.get_3d_point(cx, cy, z=1.0)
                    head_pos = [i - 0.5 for i in head_pos]

                    if head_pos:
                        x3d, y3d, z3d = head_pos
                        bottle3D = self.output_3d_port.prepare()
                        bottle3D.clear()
                        bottle3D.addFloat32(x3d)
                        bottle3D.addFloat32(y3d)
                        bottle3D.addFloat32(z3d)
                        self.output_3d_port.write()

                        print(f"[iTrackPeople] 3D head: x={x3d:.2f}, y={y3d:.2f}, z={z3d:.2f}")

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

        if self.rpc_port.write(cmd, response):
            if response.get(0).asString() == "ack":
                coords = response.get(1).asList()
                return (coords.get(0).asFloat64(),
                        coords.get(1).asFloat64(),
                        coords.get(2).asFloat64())
            else:
                print("[iTrackPeople] RPC nack.")
        else:
            print("[iTrackPeople] RPC communication failed.")
        return None

    def interruptModule(self):
        print("[iTrackPeople] Interrupting...")
        return True

    def close(self):
        print("[iTrackPeople] Closing ports and cleaning up...")
        self.input_port.close()
        self.output_port.close()
        self.output_3d_port.close()
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
