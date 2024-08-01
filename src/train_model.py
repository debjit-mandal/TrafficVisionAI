import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from utils.preprocess_data import load_and_preprocess_data

DATA_DIR = "data/GTSRB/train"
MODEL_PATH = "models/traffic_sign_model.h5"
IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 43

images, labels = load_and_preprocess_data(DATA_DIR, split="Train")
print(f"Loaded {len(images)} images.")

if len(images) == 0:
    print("No images loaded. Exiting...")
    exit()

labels = to_categorical(labels, NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
)
datagen.fit(X_train)

model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    verbose=1,
)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
