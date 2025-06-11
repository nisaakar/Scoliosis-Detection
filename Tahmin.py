import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Parametreler ===
IMG_SIZE = 128
CLASSES = ["normal", "scol"]
DATASET_DIR = "augmented_data"
TEST_DIR = "test"

model = load_model("model.keras")

# === Toplu test görseli tahmini + ön işleme ===
print("\n--- Toplu Test Sonuçları ---\n")

def preprocess_grayscale(img):
    img = cv2.equalizeHist(img)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharpened

def center_crop(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = w // 2 - min_dim // 2
    start_y = h // 2 - min_dim // 2
    return img[start_y:start_y+min_dim, start_x:start_x+min_dim]

# === Test klasörü tahminleri
print("\n--- Toplu Test Tahminleri ---\n")

for file in sorted(os.listdir(TEST_DIR)):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(TEST_DIR, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{file}: okunamadı.")
        continue
    img = center_crop(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_grayscale(img)
    img = img / 255.0
    img_input = np.expand_dims(img, axis=(0, -1))  # (1, 128, 128, 1)

    prediction = model.predict(img_input)[0][0]
    label = CLASSES[int(prediction > 0.5)]
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    print(f"{file}: {label} ({confidence:.2f}%)")
    plt.imshow(img1.squeeze(), cmap="gray")
    plt.title(f"{file} → Tahmin: {label} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
