import cv2
import os
import numpy as np
import mediapipe as mp

# Paths
DATA_DIR = "../data/processed/"
os.makedirs(DATA_DIR, exist_ok=True)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Gestures to record
GESTURES = [ "hello", "thanks", "yes","no"]  #   "hello" "thanks" "yes"You can expand this
SAMPLES_PER_GESTURE = 30  # number of recordings per gesture
FRAMES_PER_SAMPLE = 30  # frames per recording (1–2 seconds of video)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    print(f"\nRecording gesture: {gesture}")
    for sample in range(SAMPLES_PER_GESTURE):
        sequence = []
        print(f" Sample {sample + 1}/{SAMPLES_PER_GESTURE} - press 's' to start")

        # Wait until user presses 's'
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f"Press 's' to record {gesture}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        frames_recorded = 0
        while frames_recorded < FRAMES_PER_SAMPLE:
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

            if landmarks:  # only save if a hand was detected
                sequence.append(landmarks)
                frames_recorded += 1

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        sequence = np.array(sequence)
        np.save(os.path.join(DATA_DIR, f"{gesture}_{sample}.npy"), sequence)
        print(f" Saved {gesture}_{sample}.npy")

cap.release()
cv2.destroyAllWindows()
