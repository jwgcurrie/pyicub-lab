# iTrackPeople

**`iTrackPeople.py`** is a YARP-compatible module that detects the **midpoint between a person’s eyes** in real-time using [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). The midpoint coordinates are published over a YARP Bottle port and can optionally be visualized using OpenCV.

---

## Features

- Real-time processing from YARP image stream (e.g. `/grabber`)
- Computes and publishes **(u, v, z)** coordinates of the midpoint between the eyes
- Optional OpenCV visualization with red dot overlay
- Pluggable into existing iCub/YARP applications

---

## Input / Output Ports

| Port | Type | Description |
|------|------|-------------|
| `/iTrackPeople/image:i` | `yarp.ImageRgb` | Input image (typically from `/grabber`) |
| `/iTrackPeople/eyes:o`  | `yarp.Bottle`   | Output of `(u, v, z)` as floats — the midpoint between the eyes |

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

Read the output:

```bash
yarp read /iTrackPeople/eyes:o
```

---

## Output Example

```
0.483 0.276 -0.052
```

Each value corresponds to:
- **u**: normalized horizontal position (0–1)
- **v**: normalized vertical position (0–1)
- **z**: estimated depth (relative from MediaPipe)

---

## 🚼 Notes

- Only outputs values when both eyes and nose are visible (confidence > 0.5)
- Uses `model_complexity=1` for MediaPipe by default
- Add your own frame-skip or FPS control if needed for performance

