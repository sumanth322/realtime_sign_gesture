# =========================
# main.py - Real-time Sign Language Translator (Improved)
# =========================

from tensorflow.keras.models import load_model
import numpy as np
import cv2
from collections import deque

# =========================
# Load Model
# =========================
try:
    model = load_model("models/sign_model.keras")   # Preferred format
    print("✅ Loaded model from sign_model.keras")
except:
    model = load_model("models/sign_model.h5")      # Fallback legacy format
    print("✅ Loaded model from sign_model.h5")

# =========================
# Print Model Summary
# =========================
print("\n📌 Model Summary:")
model.summary()

# =========================
# Dummy Input Test
# =========================
input_shape = model.input_shape[1:]
print(f"\n📌 Model expects input shape: {input_shape}")

dummy_input = np.random.rand(1, *input_shape).astype("float32")
prediction = model.predict(dummy_input)
print("\n✅ Model is working!")
print("Prediction output shape:", prediction.shape)
print("Prediction sample:", prediction[0])

# =========================
# Real-time Webcam Capture
# =========================
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Map indices to letters
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# For smoothing predictions over frames
prediction_queue = deque(maxlen=5)  # Average over last 5 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)

    # Predict
    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    prediction_queue.append(predicted_class)

    # Use majority vote for smoothing
    smooth_prediction = max(set(prediction_queue), key=prediction_queue.count)
    predicted_letter = alphabet[smooth_prediction]

    # Display predictions and instructions
    cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to exit", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Sign Language Translator", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
