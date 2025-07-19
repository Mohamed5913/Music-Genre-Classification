import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from PIL import Image

# Force usage of GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is being used.")
    except:
        print("Could not set GPU memory growth.")
else:
    print("No GPU found. Using CPU.")

# === Constants ===
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
DATASET_PATH = 'data/genres_original/'
MFCC_PATH = 'mfcc_features/'
SPEC_PATH = 'spectrograms/'
MODEL_PATH = 'models/'
os.makedirs(MODEL_PATH, exist_ok=True)

# === Step 1: Extract MFCC features ===
def extract_mfcc_features():
    X, y = [], []
    for genre in GENRES:
        genre_dir = os.path.join(DATASET_PATH, genre)
        for filename in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, filename)
            y_audio, sr = librosa.load(file_path, duration=30)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
            mfcc = mfcc[:, :130]
            if mfcc.shape[1] < 130:
                pad_width = 130 - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
            X.append(mfcc.flatten())
            y.append(GENRES.index(genre))
    return np.array(X), np.array(y)

# === Step 2: Train Tabular Model ===
def train_tabular_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=GENRES, yticklabels=GENRES)
    plt.title("Confusion Matrix - Tabular")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# === Step 3: Generate Spectrogram Images ===
def save_spectrograms():
    for genre in GENRES:
        genre_dir = os.path.join(DATASET_PATH, genre)
        save_dir = os.path.join(SPEC_PATH, genre)
        os.makedirs(save_dir, exist_ok=True)
        for filename in os.listdir(genre_dir):
            save_path = os.path.join(save_dir, filename.replace('.wav', '.png'))
            if os.path.exists(save_path):
                continue
            file_path = os.path.join(genre_dir, filename)
            y_audio, sr = librosa.load(file_path, duration=30)
            S = librosa.feature.melspectrogram(y=y_audio, sr=sr)
            S_DB = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(3, 3))
            librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# === Step 4: Train CNN on Spectrograms ===
def train_cnn_model():
    with tf.device('/GPU:0'):
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
        train_gen = datagen.flow_from_directory(SPEC_PATH, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')
        val_gen = datagen.flow_from_directory(SPEC_PATH, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            MaxPooling2D((2,2)),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_gen, validation_data=val_gen, epochs=20)

        model.save(os.path.join(MODEL_PATH, 'cnn_model.h5'))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('CNN Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('CNN Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

# === Step 5: Transfer Learning ===
def train_transfer_learning():
    with tf.device('/GPU:0'):
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
        train_gen = datagen.flow_from_directory(SPEC_PATH, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')
        val_gen = datagen.flow_from_directory(SPEC_PATH, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        base_model.trainable = False

        model = Sequential([
            base_model,
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_gen, validation_data=val_gen, epochs=20)

        model.save(os.path.join(MODEL_PATH, 'transfer_model.h5'))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Transfer Learning Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Transfer Learning Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

# === Step 6: Predict Genre from New Audio File ===
def predict_genre_from_audio(file_path):
    y_audio, sr = librosa.load(file_path, duration=30)
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    temp_path = 'temp_spec.png'
    if not os.path.exists(temp_path):
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    img = Image.open(temp_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model(os.path.join(MODEL_PATH, 'transfer_model.h5'))
    prediction = model.predict(img_array)
    predicted_genre = GENRES[np.argmax(prediction)]
    print(f"Predicted Genre: {predicted_genre}")
    if os.path.exists(temp_path):
        os.remove(temp_path)

# === Execution Flow ===
if __name__ == "__main__":
    print("Extracting MFCC features...")
    X, y = extract_mfcc_features()
    print("Training Random Forest on MFCCs...")
    train_tabular_model(X, y)

    print("Generating spectrogram images...")
    save_spectrograms()
    print("Training CNN on spectrograms...")
    train_cnn_model()

    print("Training transfer learning model...")
    train_transfer_learning()
