# iTrackPeople

**`iTrackPeople.py`** is a YARP-compatible perception module that detects the **midpoint between a person’s eyes** in real-time using [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). The midpoint coordinates are published over YARP Bottle ports in **2D pixel space**.

---

## Features

- Real-time processing from a YARP image stream (e.g. `/grabber` or from icub.)
- Computes and publishes:
  - **2D pixel midpoint** between the eyes

---

## Input / Output Ports

| Port | Type | Description |
|------|------|-------------|
| `/iTrackPeople/image:i`   | `yarp.ImageRgb` | Input image (typically from `/grabber` or icub) |
| `/iTrackPeople/eyes:o`    | `yarp.Bottle`   | Midpoint in 2D pixel space: `(u, v)` |
---

## Dependencies

- Python 3
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- MediaPipe (`mediapipe`)

Install them with:

```bash
pip install -r requirements.txt
```

---

## Usage
To launch with default period of 0.1 s

```bash
python3 iTrackPeople.py
```
or to set period

```bash
python3 iTrackPeople.py --period 0.1
```

Then connect your camera/image source:

```bash
yarp connect /grabber /iTrackPeople/image:i
```

And read the outputs:

```bash
yarp read /anyname --from /iTrackPeople/eyes:o
```

---

## Output Examples

**2D Pixel Output (`/iTrackPeople/eyes:o`)**:
```
327 198
```

## Notes
- Only outputs values when both eyes are confidently detected (visibility > 0.5)
- Uses `model_complexity=1` for MediaPipe Pose
