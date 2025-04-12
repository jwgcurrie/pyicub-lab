import cv2
import numpy as np
from keras.models import load_model

# ----------------------- CONFIG -----------------------

DNN_PROTO_PATH = "models/deploy.prototxt.txt"
DNN_MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"
EMOTION_MODEL_PATH = "mini_XCEPTION.h5"
CONF_THRESHOLD = 0.6

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_COLORS = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Sad': (255, 0, 0),
    'Surprise': (0, 165, 255),
    'Neutral': (128, 128, 128)
}

# ----------------------- INIT -----------------------

# Load models
face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)

# ----------------------- FUNCTIONS -----------------------

def detect_faces(frame, conf_threshold=CONF_THRESHOLD):
    """Detect faces using OpenCV DNN."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x, y = max(0, x1), max(0, y1)
            w_box, h_box = min(w, x2) - x, min(h, y2) - y
            faces.append((x, y, w_box, h_box))
    return faces

def predict_emotion(face_img):
    """Predict emotion from face image."""
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    preds = emotion_model.predict(face_img, verbose=0)
    return EMOTION_LABELS[np.argmax(preds)]

def draw_emotion(frame, x, y, w, h, emotion):
    """Draw bounding box and label for detected emotion."""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))  # fallback = white
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)

# ----------------------- MAIN -----------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            emotion = predict_emotion(face_gray)
            draw_emotion(frame, x, y, w, h, emotion)

        cv2.imshow("Affect Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------- RUN -----------------------

if __name__ == "__main__":
    main()
