import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# === Parametreler ===
IMG_SIZE = 128
CLASSES = ["normal", "scol"]
DATASET_DIR = "augmented_data"

# === Görsel iyileştirme ===
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

# === Verileri yükle ===
X, y = [], []
for label in CLASSES:
    class_path = os.path.join(DATASET_DIR, label)
    images = glob(os.path.join(class_path, "*.jpg")) + glob(os.path.join(class_path, "*.png"))
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_grayscale(img)
        img = img / 255.0
        X.append(img)
        y.append(CLASSES.index(label))

X = np.array(X)
X = np.expand_dims(X, axis=-1)
y = np.array(y)

print("Veri şekli:", X.shape)
print("Sınıf dağılımı:", np.bincount(y))

# === Eğitim/Test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Augmentasyon
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# === Class weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# === Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# === Eğitim
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=25,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# === Kaydet
model.save("model.keras")
print("\nModel kaydedildi: model.keras")

# === Accuracy / Loss Grafiği
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Eğitim Doğruluğu")
plt.plot(history.history['val_accuracy'], label="Doğrulama Doğruluğu")
plt.title("Model Doğruluğu")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Eğitim Kaybı")
plt.plot(history.history['val_loss'], label="Doğrulama Kaybı")
plt.title("Model Kayıp Değeri")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# === Confusion Matrix + Rapor
y_pred_probs = model.predict(X_test)
y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
print("\nSınıflandırma Raporu:\n")
print(classification_report(y_test, y_pred_classes, target_names=CLASSES))
