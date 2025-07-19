import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import os
import tempfile

# === Load Genre Labels ===
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# === Load Model ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'transfer_model.h5')
model = load_model(MODEL_PATH)

# === Prediction Logic ===
def predict_genre(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        temp_img_path = os.path.join(tempfile.gettempdir(), "temp_spec.png")
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(temp_img_path).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = prediction[0][predicted_index]
        return predicted_genre, confidence

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None

# === File Upload Logic ===
def choose_file():
    filetypes = (("Audio files", "*.wav *.mp3"), ("All files", "*.*"))
    filepath = filedialog.askopenfilename(title="Choose an audio file", filetypes=filetypes)
    if not filepath:
        return

    result_label.config(text="🔍 Predicting...")
    root.update()

    genre, confidence = predict_genre(filepath)
    if genre:
        result_label.config(text=f"🎧 Genre: {genre.capitalize()} \nConfidence: {confidence:.2%}")
    else:
        result_label.config(text="Prediction failed.")

# === GUI Layout ===
root = tk.Tk()
root.title("🎵 Music Genre Classifier")
root.geometry("550x350")
root.resizable(False, False)
root.configure(bg="#f0f2f5")

# === Title ===
title = tk.Label(root, text="🎵 Music Genre Classifier", font=("Helvetica", 20, "bold"), bg="#f0f2f5", fg="#333")
title.pack(pady=30)

# === Styled Upload Button ===
style = ttk.Style()
style.configure("Custom.TButton",
                font=("Helvetica", 14),
                padding=10)

upload_btn = ttk.Button(root, text="Upload .wav or .mp3 File", style="Custom.TButton", command=choose_file)
upload_btn.pack(pady=20)

# === Result Label ===
result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f2f5", fg="#444")
result_label.pack(pady=20)

# === Run the App ===
root.mainloop()