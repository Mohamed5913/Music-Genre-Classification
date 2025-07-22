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

# Load Genre Labels
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'transfer_model.h5')
model = load_model(MODEL_PATH)

# Prediction Logic
def predict_genre(file_path):
    y_audio, sr = librosa.load(file_path, duration=30)
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    temp_path = 'temp_spec.png'
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = Image.open(temp_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model(MODEL_PATH)
    prediction = model.predict(img_array)[0]

    top_n = 3
    top_indices = prediction.argsort()[-top_n:][::-1]
    top_genres = [(GENRES[i], float(prediction[i]) * 100) for i in top_indices]

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return top_genres


# File Upload Logic
def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        try:
            top_genres = predict_genre(file_path)

            result_text = "Top 3 Predictions:\n\n"
            for genre, confidence in top_genres:
                result_text += f"{genre.capitalize()}: {confidence:.2f}%\n"

            result_label.config(text=result_text)

        except Exception as e:
            result_label.config(text=f"❌ Error: {str(e)}")

# GUI
root = tk.Tk()
root.title("🎵 Music Genre Classifier")
root.resizable(False, False)
root.configure(bg="#f0f2f5")
window_width = 550
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y- 50}")

title = tk.Label(root, text="🎵 Music Genre Classifier", font=("Helvetica", 20, "bold"), bg="#f0f2f5", fg="#333")
title.pack(pady=30)

style = ttk.Style()
style.configure("Custom.TButton", font=("Helvetica", 14), padding=10)

upload_btn = ttk.Button(root, text="Upload .wav or .mp3 File", style="Custom.TButton", command=choose_file)
upload_btn.pack(pady=20)

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f2f5", fg="#444", justify="left")
result_label.pack(pady=20)

root.mainloop()
