# =========================
# main.py - Real-time Sign Language Translator with Gesture Streaming
# =========================

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
import socketio

# =========================
# SocketIO setup
# =========================
SIGNALING_SERVER = "http://localhost:5000"
ROOM = "testroom"

sio = socketio.Client()
sio.connect(SIGNALING_SERVER)
sio.emit("join", {"room": ROOM})

# =========================
# Load LSTM Gesture Model
# =========================
try:
    model = load_model("models/sign_model_final.keras")
    print("✅ Loaded model from sign_model_final.keras")
except:
    model = load_model("models/sign_model_final.h5")
    print("✅ Loaded model from sign_model_final.h5")

gestures = ['hello', 'no', 'thanks', 'yes']

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Webcam & Sequence Setup
# =========================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=30)  # store last 30 frames of landmarks

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # Append landmarks or zeros if no hand detected
    if landmarks:
        sequence.append(landmarks)
    else:
        sequence.append([0]*63)

    gesture_label = ""
    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)  # (1, 30, 63)
        pred = model.predict(input_data, verbose=0)
        gesture_class = np.argmax(pred)
        score = float(np.max(pred))
        gesture_label = gestures[gesture_class]

        # Send gesture via SocketIO to other peer
        sio.emit("gesture", {"room": ROOM, "label": gesture_label, "score": score})
        print(f"➡️ Sent gesture: {gesture_label} ({score:.2f})")

        # Display gesture on local frame
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Instructions
    cv2.putText(frame, "Press 'q' to exit", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Gesture Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
sio.disconnect()
