# FocusSense/train_focus_model.py
"""
Training script for FocusSense – Eye & Head Pose Detection for Study Concentration.

Model:
- TimeDistributed Dense layers (CNN-like feature encoder)
- LSTM for temporal learning
- Binary classification: Focused / Distracted
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# -----------------------------
# Configuration
# -----------------------------
SEQ_LEN = 30
FEATURE_DIM = 9
BATCH_SIZE = 16
EPOCHS = 20
MODEL_PATH = "focussense_model.h5"
DATA_DIR = "data"


# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(data_dir):
    X, y = [], []

    for fname in os.listdir(data_dir):
        if not fname.endswith(".npy"):
            continue

        seq = np.load(os.path.join(data_dir, fname))

        # Expected shape: (1, 30, 9)
        if seq.shape != (1, SEQ_LEN, FEATURE_DIM):
            continue

        X.append(seq[0])  # remove extra dimension

        if "focused" in fname.lower():
            y.append(1)
        else:
            y.append(0)

    return np.array(X), np.array(y)


# -----------------------------
# Build CNN + LSTM model
# -----------------------------
def build_model():
    model = Sequential()
    model.add(Input(shape=(SEQ_LEN, FEATURE_DIM)))

    model.add(TimeDistributed(Dense(64, activation="relu")))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(32, activation="relu")))

    model.add(LSTM(128, return_sequences=False))

    model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Main training pipeline
# -----------------------------
if __name__ == "__main__":

    print("Loading FocusSense dataset...")
    X, y = load_dataset(DATA_DIR)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) == 0:
        raise RuntimeError("No training data found. Please collect data first.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building CNN + LSTM model...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
    ]

    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
