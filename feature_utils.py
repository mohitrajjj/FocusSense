# focussense/feature_utils.py
"""
Feature extraction utilities for FocusSense.

This module converts MediaPipe FaceMesh landmarks into
numerical feature vectors suitable for CNN + LSTM models.

We reuse landmarks already computed in main.py.
"""

import numpy as np


# ---- Selected landmark indices (MediaPipe Face Mesh) ----
# These are stable and commonly used indices
LEFT_EYE = [33, 133]        # left eye corners
RIGHT_EYE = [362, 263]     # right eye corners
NOSE_TIP = 1               # nose tip landmark


def _avg_landmark(landmarks, indices, img_w, img_h):
    """
    Compute average (x, y) position of given landmark indices.
    Normalized to image size.
    """
    xs = [landmarks[i].x * img_w for i in indices]
    ys = [landmarks[i].y * img_h for i in indices]
    return np.mean(xs) / img_w, np.mean(ys) / img_h


def extract_focussense_features(face_landmarks, img_w, img_h):
    """
    Extract FocusSense feature vector from MediaPipe landmarks.

    Parameters:
    - face_landmarks : MediaPipe face landmarks object
    - img_w, img_h   : image width and height

    Returns:
    - feature_vector : numpy array of shape (N,)
      or None if landmarks are invalid
    """

    if face_landmarks is None:
        return None

    lm = face_landmarks.landmark

    # ---- Eye centers ----
    lx, ly = _avg_landmark(lm, LEFT_EYE, img_w, img_h)
    rx, ry = _avg_landmark(lm, RIGHT_EYE, img_w, img_h)

    # ---- Nose tip ----
    nx = lm[NOSE_TIP].x
    ny = lm[NOSE_TIP].y

    # ---- Eye midpoint ----
    mx = (lx + rx) / 2
    my = (ly + ry) / 2

    # ---- Head direction approximation ----
    # Vector from nose tip to eye midpoint
    head_vec_x = mx - nx
    head_vec_y = my - ny

    # ---- Eye distance (proxy for openness / face distance) ----
    eye_distance = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

    # ---- Final feature vector ----
    feature_vector = np.array([
        lx, ly,             # left eye center
        rx, ry,             # right eye center
        nx, ny,             # nose position
        head_vec_x, head_vec_y,
        eye_distance
    ], dtype=np.float32)

    return feature_vector

def normalize(features):
    features = np.array(features, dtype=np.float32)

    # Example normalization (adjust per feature meaning)
    features[0:2] /= 640.0     # x coordinates
    features[2:4] /= 480.0     # y coordinates
    features[4:7] /= 90.0      # pitch, yaw, roll
    features[7:] /= 10.0       # small ratios / movements

    return features

    return normalize(features)

