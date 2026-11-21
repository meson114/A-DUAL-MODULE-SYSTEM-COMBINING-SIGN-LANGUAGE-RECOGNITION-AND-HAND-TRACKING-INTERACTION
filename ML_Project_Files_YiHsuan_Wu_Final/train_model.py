"""
Module 2: Model Architecture & Training Pipeline
Implements the Conv1D + LSTM hybrid architecture for 64-class sign language recognition.
Features Dropout regularization, EarlyStopping callback, and validation logging.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 64
SEQUENCE_LENGTH = 60
FEATURE_DIM = 63

def build_conv1d_lstm_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    return model

if __name__ == "__main__":
    print("[INFO] Initializing Conv1D + LSTM Model...")
    model = build_conv1d_lstm_model()
    model.summary()
