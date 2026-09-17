# FocusSense/sequence_buffer.py
"""
Sequence buffer for FocusSense.

Maintains a fixed-length sliding window of feature vectors
to be used as input for the LSTM model.
"""

import numpy as np
from collections import deque


class SequenceBuffer:
    def __init__(self, seq_len=30, feature_dim=9):
        """
        Parameters:
        - seq_len     : number of frames per sequence (e.g., 30)
        - feature_dim : length of feature vector per frame
        """
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=seq_len)


    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()

    def add(self, feature_vector):
        """
        Add a new feature vector to the buffer.

        Parameters:
        - feature_vector : numpy array of shape (feature_dim,)
        """
        if feature_vector is None:
            return

        if len(feature_vector) != self.feature_dim:
            return

        self.buffer.append(feature_vector)

    def is_full(self):
        """Check if buffer has enough frames for prediction."""
        return len(self.buffer) == self.seq_len

    def get_sequence(self):
        """
        Get buffered sequence as LSTM-ready input.

        Returns:
        - numpy array of shape (1, seq_len, feature_dim)
          or None if buffer not full
        """
        if not self.is_full():
            return None

        seq = np.array(self.buffer, dtype=np.float32)
        return np.expand_dims(seq, axis=0)
