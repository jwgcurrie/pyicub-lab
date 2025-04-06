# iTrackPeople

**`iTrackPeople.py`** is a YARP-compatible perception module that detects the **midpoint between a person’s eyes** in real-time using [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). The midpoint coordinates are published over YARP Bottle ports in both **2D pixel space** and **3D real-world space** using iCub’s internal coordinate frame via `iKinGazeCtrl`.

---

## Features

- Real-time processing from a YARP image stream (e.g. `/grabber`)
- Computes and publishes:
  - **2D pixel midpoint** between the eyes
  - **3D head position** using RPC call `get 3D mono` from `/iKinGazeCtrl/rpc`
- Optional OpenCV visualization with red dot overlay
- Fully pluggable into iCub/YARP pipelines

---

## Input / Output Ports

| Port | Type | Description |
|------|------|-------------|
| `/iTrackPeople/image:i`   | `yarp.ImageRgb` | Input image (typically from `/grabber`) |
| `/iTrackPeople/eyes:o`    | `yarp.Bottle`   | Midpoint in 2D pixel space: `(u, v)` |
| `/iTrackPeople/head3D:o`  | `yarp.Bottle`   | 3D position in iCub’s reference frame: `(x, y, z)` in **meters** |

---

## Dependencies

- Python 3
- OpenCV (`opencv-python`)
- NumPy
- MediaPipe (`mediapipe`)
- YARP Python bindings

Install them with:

```bash
pip install opencv-python mediapipe numpy
```

---

## Usage

Run the module with optional display:

```bash
python3 iTrackPeople.py --display
```

Or headless:

```bash
python3 iTrackPeople.py
```

Then connect your camera/image source:

```bash
yarp connect /grabber /iTrackPeople/image:i
```

And read the outputs:

```bash
yarp read /anyname --from /iTrackPeople/eyes:o
yarp read /anyname --from /iTrackPeople/head3D:o
```

---

## Output Examples

**2D Pixel Output (`/iTrackPeople/eyes:o`)**:
```
327 198
```

**3D Head Position (`/iTrackPeople/head3D:o`)**:
```
0.009 0.933 0.979
```

Each 3D coordinate is in **meters**, relative to iCub’s internal reference frame.

---

## Notes

- 3D position is retrieved via `get 3D mono ('left' u v 1.0)` from `/iKinGazeCtrl/rpc`
- Uses a fixed depth of 1.0m for mono back-projection
- Only outputs values when both eyes are confidently detected (visibility > 0.5)
- Uses `model_complexity=1` for MediaPipe Pose
