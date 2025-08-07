"""
Rock-Paper-Scissors Image Classifier with GUI
Trains a CNN model using image data augmentation, callbacks, and provides
a simple GUI for inference using a saved model.
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk, Label, Button, PhotoImage
from PIL import Image, ImageTk
import numpy as np

# --- Parameters ---
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
dataset_path = "data/kaggle_dataset/rps-cv-images"
checkpoint_path = "checkpoints/rps_best_model.h5"
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
EPOCHS = 20

# --- Data Augmentation and Generators ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- Model Definition ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, save_best_only=True),
    TensorBoard(log_dir=log_dir)
]

# --- Model Training ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# --- Save Final Model ---
model.save("rps_model_final.h5")

# ========== GUI for Inference ==========

def load_and_predict(image_path):
    """Load and preprocess the image, then return prediction label."""
    img = Image.open(image_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model_loaded.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    return class_names[np.argmax(prediction)]

def open_file():
    """Open image file and update GUI with prediction."""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load image and show
    img = Image.open(file_path).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Predict and display result
    result = load_and_predict(file_path)
    prediction_label.config(text=f"Prediction: {result}")

# Load saved model
model_loaded = load_model("rps_model_final.h5")

# --- GUI Setup ---
root = Tk()
root.title("Rock Paper Scissors Classifier")
root.geometry("400x400")

Label(root, text="Upload an image of Rock, Paper, or Scissors").pack(pady=10)
Button(root, text="Choose Image", command=open_file).pack(pady=10)

image_label = Label(root)
image_label.pack()

prediction_label = Label(root, text="", font=("Arial", 14))
prediction_label.pack(pady=10)

root.mainloop()
