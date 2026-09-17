# FocusSense/focus_inference.py
"""
Real-time inference module for FocusSense.

Loads trained CNN + LSTM model and predicts
Focused / Distracted state from feature sequences.
Also sends live results to dashboard (REST).
"""

import numpy as np
from tensorflow.keras.models import load_model
from sequence_buffer import SequenceBuffer
import os
import requests
import time


class FocusSenseInference:
    def __init__(
        self,
        model_path="focussense_model.h5",
        seq_len=30,
        feature_dim=9,
        threshold=0.6,
        dashboard_url="http://127.0.0.1:5000/update"
    ):
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print("✅ FocusSense model loaded")
        else:
            self.model = None
            print("⚠️ FocusSense model not found. Running in DATA COLLECTION mode.")

        # Internal buffer
        self.buffer = SequenceBuffer(seq_len, feature_dim)

        self.threshold = threshold
        self.last_prediction = None
        self.last_score = 0

        # Dashboard config
        self.dashboard_url = dashboard_url
        self.last_sent_time = 0  # rate limit

    def reset(self):
        self.buffer.reset()
        self.last_prediction = None
        self.last_score = 0

    def send_to_dashboard(self, prediction, score):
        """
        Sends result to dashboard (non-blocking)
        """
        now = time.time()

        # ⛔ Send at most once per second
        if now - self.last_sent_time < 1:
            return

        payload = {
            "status": prediction,
            "score": score / 100.0
        }

        try:
            requests.post(
                self.dashboard_url,
                json=payload,
                timeout=0.2
            )
            self.last_sent_time = now
        except:
            pass  # NEVER break inference

    def update(self, feature_vector):
        """
        Parameters:
        - feature_vector : numpy array of shape (feature_dim,)

        Returns:
        - prediction ("Focused"/"Distracted"/None)
        - score (0–100 or None)
        """

        # Add one frame
        self.buffer.add(feature_vector)

        # Not enough frames yet
        if not self.buffer.is_full():
            return None, None

        # Shape: (1, 30, 9)
        seq = self.buffer.get_sequence()

        # Predict
        prob = float(self.model.predict(seq, verbose=0)[0][0])
        score = int(prob * 100)

        if prob >= self.threshold:
            prediction = "Focused"
        else:
            prediction = "Distracted"

        self.last_prediction = prediction
        self.last_score = score

        # 🔥 SEND TO DASHBOARD
        self.send_to_dashboard(prediction, score)

        return prediction, score
