import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

model = tf.keras.models.load_model("saved_model/cifar10_model.h5")
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

root = tk.Tk()
root.title("🧠 CIFAR-10 Image Classifier")
root.geometry("700x700")
root.configure(bg="white")

image_label = tk.Label(root, bg="white")
image_label.pack(pady=20)

prediction_label = tk.Label(root, text="", font=("Helvetica", 16), bg="white")
prediction_label.pack()

current_index = [0]
last_image_pil = None  # for saving

def show_image(index):
    global last_image_pil
    img = x_test[index]
    img_uint8 = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, 'RGB')
    img_resized = img_pil.resize((512, 512), Image.LANCZOS)
    enhancer = ImageEnhance.Sharpness(img_resized)
    img_sharp = enhancer.enhance(1.5)
    img_tk = ImageTk.PhotoImage(img_sharp)

    image_label.configure(image=img_tk)
    image_label.image = img_tk
    last_image_pil = img_sharp

def predict_image():
    index = current_index[0]
    img = x_test[index]
    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    # Get top-3 predictions
    top3_idx = prediction.argsort()[-3:][::-1]
    top3_text = "\n".join(
        [f"{labels[i]}: {prediction[i]*100:.2f}%" for i in top3_idx])

    prediction_label.config(text=top3_text)

def next_image():
    current_index[0] = (current_index[0] + 1) % len(x_test)
    show_image(current_index[0])
    prediction_label.config(text="")

def save_image():
    if last_image_pil:
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if filename:
            last_image_pil.save(filename)

btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict", command=predict_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Next Image", command=next_image, font=("Arial", 12), bg="#2196F3", fg="white", padx=10).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Save Image", command=save_image, font=("Arial", 12), bg="#f57c00", fg="white", padx=10).grid(row=0, column=2, padx=10)

show_image(current_index[0])
root.mainloop()
