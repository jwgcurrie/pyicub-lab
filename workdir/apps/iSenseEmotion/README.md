# iSenseEmotion

**iSenseEmotion** is a lightweight real-time facial emotion recognition module built with OpenCV and a pre-trained deep learning model. It detects faces using OpenCV's DNN backend and classifies emotional affect using the `mini_XCEPTION` model trained on the FER-2013 dataset.


---

## Features

- Real-time face detection using OpenCV DNN (SSD + ResNet10)
- Emotion classification with mini_XCEPTION
- Emotion-based bounding box coloring

---


## Getting Started

### 🔧 Dependencies

Install required packages:
```bash
pip install opencv-python keras tensorflow numpy
```

### Download Required Files

1. **mini_XCEPTION.h5**  
   Download from [oarriaga's repo](https://github.com/oarriaga/face_classification)

2. **Face Detection Model**  
   - [deploy.prototxt.txt](https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/blob/master/deploy.prototxt.txt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

Place these inside the `models/` directory.

---

## Run the Module

```bash
python iSenseEmotion.py
```

Press `q` to exit the window.

---

## Emotion Labels

This module recognizes the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Each is color-coded based on common affective associations.