# feature_extractor.py
import numpy as np

def extract_features(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    pitch = (nose.y - 0.5) * 100
    yaw = (nose.x - 0.5) * 100
    roll = 0.0

    left_dx = left_eye.x - nose.x
    left_dy = left_eye.y - nose.y
    right_dx = right_eye.x - nose.x
    right_dy = right_eye.y - nose.y

    blink_ratio = abs(left_dy - right_dy)
    blink = 1 if blink_ratio > 0.015 else 0

    return np.array([
        pitch, yaw, roll,
        left_dx, left_dy,
        right_dx, right_dy,
        blink_ratio, blink
    ], dtype=np.float32)
