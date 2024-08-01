import os

import cv2
import numpy as np
import pandas as pd


def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image / 255.0
    return image


def load_and_preprocess_data(data_dir, split="Train"):
    images = []
    labels = []

    csv_file = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        img_path = os.path.join(data_dir, row["Path"])
        print(f"Trying to load: {img_path}")
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue

        image = preprocess_image(image)
        images.append(image)
        labels.append(row["ClassId"])

    return np.array(images), np.array(labels)
