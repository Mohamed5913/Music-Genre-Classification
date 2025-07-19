import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
from pydub import AudioSegment
import gdown

# === Load Model from Google Drive if not present ===
def download_model():
    model_url = "https://drive.google.com/uc?id=1MZam4JfHtEtWTbF2cTACwxzXxSl5A7Tn"  # public file ID for transfer_model.h5
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "transfer_model.h5")
    if not os.path.exists(model_path):
        with st.spinner("📦 Downloading model from Google Drive..."):
            gdown.download(model_url, model_path, quiet=False)
    return load_model(model_path)

model = download_model()

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_DESCRIPTIONS = {
    'blues': "A genre known for its melancholic sound, often using guitar and harmonica.",
    'classical': "Orchestral music composed from historical eras such as Baroque and Romantic.",
    'country': "American folk music characterized by ballads and dance tunes.",
    'disco': "Dance music from the 1970s with upbeat tempos and synths.",
    'hiphop': "Urban music genre with rhythmic vocals and beats.",
    'jazz': "Improvisational genre with complex chords and instrumental solos.",
    'metal': "Heavy guitar riffs and intense vocals, often with themes of power.",
    'pop': "Mainstream popular music with catchy melodies.",
    'reggae': "Jamaican genre with offbeat rhythms and socially conscious lyrics.",
    'rock': "A broad genre centered on amplified instruments and rebellious themes."
}

# === Helper: Convert MP3 to WAV ===
def convert_to_wav(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file_path = tmp.name
        if uploaded_file.name.endswith(".mp3"):
            audio = AudioSegment.from_file(uploaded_file, format="mp3")
            audio.export(file_path, format="wav")
        else:
            tmp.write(uploaded_file.read())
        return file_path

# === Helper: Predict Genre ===
@st.cache_data(show_spinner=False)
def predict_genre(audio_file):
    y_audio, sr = librosa.load(audio_file, duration=30)
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = librosa.util.fix_length(S_DB, size=128, axis=1)
    if S_DB.shape[0] < 128:
        S_DB = np.pad(S_DB, ((0, 128 - S_DB.shape[0]), (0, 0)))
    S_DB = np.stack((S_DB,)*3, axis=-1)
    S_DB = np.expand_dims(S_DB, axis=0)
    S_DB = S_DB / 255.0
    prediction = model.predict(S_DB)
    predicted_index = np.argmax(prediction)
    return GENRES[predicted_index], prediction[0][predicted_index]

# === Streamlit UI ===
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎼 Music Genre Classification App")
st.markdown("Upload a music clip (.wav or .mp3), and the model will predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner('Analyzing the audio...'):
        audio_path = convert_to_wav(uploaded_file)
        genre, confidence = predict_genre(audio_path)
        st.success(f"🎧 Predicted Genre: **{genre.capitalize()}**")
        st.write(f"🧠 Confidence: **{confidence * 100:.2f}%**")
        st.markdown(f"📝 {GENRE_DESCRIPTIONS[genre]}")
        os.remove(audio_path)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow, Librosa, and Streamlit.")