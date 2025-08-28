import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Load trained model
model = tf.keras.models.load_model("rps_model.h5")

# Class labels (must match training order)
class_names = ["Rock", "Paper", "Scissors"]

def predict_image(img_path):
    """Predict class for a single image path."""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

def open_file():
    """Open file dialog and predict image class."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)

        panel.config(image=img_tk)
        panel.image = img_tk

        label, conf = predict_image(file_path)
        result_label.config(text=f"Prediction: {label} ({conf:.2f}%)")

# GUI Setup
root = tk.Tk()
root.title("Rock-Paper-Scissors Classifier")

panel = Label(root)  
panel.pack(pady=10)

btn = Button(root, text="Choose Image", command=open_file)
btn.pack(pady=10)

result_label = Label(root, text="Prediction: None", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
