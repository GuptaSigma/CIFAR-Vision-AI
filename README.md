# 🧠 CIFAR-10 Image Classifier GUI

A full-featured **AI desktop application** that classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) — all wrapped inside a smooth **Tkinter GUI**.

Built for deep learning enthusiasts, hackathon demos, and education!

---

## 🚀 Features

| Feature                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| 🔄 Random Image           | Loads a random CIFAR-10 test image on button click                          |
| 📊 Top-3 Predictions      | Displays top 3 predicted classes with confidence                            |
| 📈 Prediction Bar Chart   | Shows class probabilities (0–1.0) for all 10 classes using Matplotlib       |
| 📁 Save Image             | Save the image with predicted label to disk                                 |
| 🖼️ Upload Custom Image    | Upload your own 32x32 `.png` or `.jpg` image and classify it                |
| 🌙 Dark Mode Toggle       | Instantly switch between Light and Dark GUI themes                          |
| 🧪 Test All Images Button | Evaluate model on complete CIFAR-10 test set and show overall accuracy      |

---

## 📂 Project Structure

CIFAR10-GUI-Classifier/
├── gui_app.py # Tkinter GUI main file
├── model.py # CNN model training script
├── predict.py # Model loading & prediction
├── utils.py # Utility functions (preprocessing etc.)
├── saved_model/
│ └── cifar10_model.h5 # Trained model (auto-generated)
├── assets/ # Optional: icons, themes, demo images
└── README.md # This file

yaml
Copy
Edit

---

## 📦 Requirements

Install dependencies:

```bash
pip install tensorflow numpy matplotlib opencv-python pillow
If using virtual env:

bash
Copy
Edit
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
🧠 How It Works
Model is trained using model.py on CIFAR-10 dataset

Predictions are made using predict.py and utils.py

gui_app.py launches the user-friendly interface to interact with the model

📷 Sample Screenshot


🏁 To Run the App
bash
Copy
Edit
python gui_app.py
If model not found, app will auto-train and save it.

📈 Accuracy (Example)
Dataset	Accuracy
CIFAR-10	82–90%
Custom Test	Varies

📌 CIFAR-10 Classes
airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

🧪 Future Ideas
Export predictions to CSV

Deploy GUI as a web app (e.g. Streamlit or Gradio)

Add Grad-CAM visualization for explainability

👨‍💻 Created By
🚀 Sagar (group leader)
👨‍💻ankur
👨‍💻sumya

