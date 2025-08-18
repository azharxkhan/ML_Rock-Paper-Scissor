import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========== Setup ==========
dataset_path = "/home/wtc/ML_Rock-Paper-Scissor/data/datasets/drgfreeman/rockpaperscissors/versions/2/rps-cv-images"
model_path = "rps_model.h5"
img_height, img_width = 150, 150
batch_size = 32
epochs = 10

# ========== Load and Preprocess Data ==========
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_indices = train_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# ========== Build Model ==========
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========== Custom Visualizer Callback ==========
class TrainingVisualizer(Callback):
    def __init__(self, val_generator, idx_to_class):
        super().__init__()
        self.val_generator = val_generator
        self.idx_to_class = idx_to_class
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        # Store logs
        self.history["loss"].append(logs["loss"])
        self.history["val_loss"].append(logs["val_loss"])
        self.history["accuracy"].append(logs["accuracy"])
        self.history["val_accuracy"].append(logs["val_accuracy"])

        # Plot training curves
        plt.figure(figsize=(10,4))

        # Loss subplot
        plt.subplot(1,2,1)
        plt.plot(self.history["loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()

        # Accuracy subplot
        plt.subplot(1,2,2)
        plt.plot(self.history["accuracy"], label="Train Acc")
        plt.plot(self.history["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        # Confusion Matrix on small validation batch
        x_val, y_val = next(self.val_generator)
        y_pred = self.model.predict(x_val, verbose=0)
        cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.idx_to_class.values()))
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix (Epoch {epoch+1})")
        plt.show(block=False)
        plt.pause(1)
        plt.close()

# ========== Train Model with Visualizer ==========
print("🧠 Training model...")
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True),
    TrainingVisualizer(val_generator, idx_to_class)
]

model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)
model.save(model_path)
print(f"✅ Model saved to {model_path}")
