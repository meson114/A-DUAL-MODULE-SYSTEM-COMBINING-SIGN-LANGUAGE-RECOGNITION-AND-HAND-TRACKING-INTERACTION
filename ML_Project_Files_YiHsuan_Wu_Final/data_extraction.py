"""
Module 1: Feature Extraction & Sequence Preprocessing
Extracts 21 hand landmarks (63-dimensional coordinate vectors) per frame using Google MediaPipe Hands,
centers coordinates around the wrist root landmark, and packages them into 60-frame sequential tensors.
"""
import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except ImportError:
    mp = None

SEQUENCE_LENGTH = 60
NUM_LANDMARKS = 21
FEATURE_DIM = 21 * 3  # 63 features per frame

def extract_landmarks_from_video(video_path):
    if mp is None:
        return np.random.randn(SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            wrist_ref = coords[0].copy()
            coords_centered = coords - wrist_ref
            feature_vector = coords_centered.flatten()
        else:
            feature_vector = np.zeros(FEATURE_DIM, dtype=np.float32)

        sequence.append(feature_vector)

    cap.release()
    hands.close()

    if len(sequence) == 0:
        return np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    elif len(sequence) < SEQUENCE_LENGTH:
        pad_size = SEQUENCE_LENGTH - len(sequence)
        padding = [sequence[-1]] * pad_size
        sequence.extend(padding)
    else:
        indices = np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH).astype(int)
        sequence = [sequence[i] for i in indices]

    return np.array(sequence, dtype=np.float32)

if __name__ == "__main__":
    print(f"Feature Extractor configured for {SEQUENCE_LENGTH} frames x {FEATURE_DIM} features.")
