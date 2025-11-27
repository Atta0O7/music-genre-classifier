# app.py
# Streamlit web application for Music Genre Classification using a CNN model.

import streamlit as st
import numpy as np
import librosa
import joblib

# ---- Safe TensorFlow Import ----
try:
    import tensorflow as tf
    TF_IMPORT_OK = True
    TF_ERROR_MSG = ""
except Exception as e:
    tf = None
    TF_IMPORT_OK = False
    TF_ERROR_MSG = repr(e)


# ---- Cached Loader: Model + Scaler + Genre Mapping ----
@st.cache_data
def load_model_and_scaler():
    """
    Loads the pre-trained Keras CNN model and the StandardScaler object.
    Cached so it only runs once and reuses results.
    """
    if not TF_IMPORT_OK:
        st.error(
            "TensorFlow is not available, so the CNN model cannot be loaded. "
            "Please fix the TensorFlow installation and restart the app."
        )
        st.stop()

    try:
        # Load the Keras model (no need to compile for inference)
        model = tf.keras.models.load_model("music_genre_cnn.h5", compile=False)

        # Load the scaler object
        scaler = joblib.load("scaler.joblib")

        # Label index -> genre name
        genre_mapping = {
            0: "blues",
            1: "classical",
            2: "country",
            3: "disco",
            4: "hiphop",
            5: "jazz",
            6: "metal",
            7: "pop",
            8: "reggae",
            9: "rock",
        }

        return model, scaler, genre_mapping

    except FileNotFoundError as e:
        st.error(
            f"Error loading model or scaler file: {e}. "
            "Please ensure 'music_genre_cnn.h5' and 'scaler.joblib' are in the project folder."
        )
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model/scaler: {e}")
        st.stop()


# ---- Feature Extraction from Audio ----
def extract_features(audio_file, sample_rate=22050, n_mfcc=13, n_chroma=12):
    """
    Extracts a 28-dimensional feature vector from a single audio file.

    Returns:
        np.array of shape (28,) or None on error.
    """
    try:
        # Load audio (Streamlit's UploadedFile is a file-like object)
        y, sr = librosa.load(audio_file, sr=sample_rate, duration=30)

        # MFCCs (13)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral Centroid (1)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)

        # Spectral Rolloff (1)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_roll_mean = np.mean(spec_roll)

        # Zero-Crossing Rate (1)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Concatenate to get 28 features: 13 + 12 + 1 + 1 + 1
        features = np.concatenate(
            [
                mfccs_mean,
                chroma_mean,
                np.array([spec_cent_mean]),
                np.array([spec_roll_mean]),
                np.array([zcr_mean]),
            ]
        )

        return features

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None


# ---- Full Prediction Pipeline ----
def predict_genre(audio_file):
    """
    The main prediction pipeline. It takes an audio file, processes it,
    and returns the predicted genre name (string).

    Args:
        audio_file: The file-like object from st.file_uploader.

    Returns:
        str: The predicted genre name or an error message.
    """
    # Step 1: Load model, scaler and mapping (cached)
    model, scaler, genre_mapping = load_model_and_scaler()

    # Step 2: Extract features -> (28,)
    features = extract_features(audio_file)

    # If feature extraction failed
    if features is None:
        return "Error: Could not process audio file. Please try a different file."

    # Step 3: Scale features (scaler expects shape (n_samples, n_features))
    try:
        features_reshaped = features.reshape(1, -1)  # (1, 28)
        features_scaled = scaler.transform(features_reshaped)  # (1, 28)
    except Exception as e:
        st.error(f"Error during feature scaling: {e}")
        return "Error: Feature scaling failed."

    # Step 4: Prepare input for CNN: (1, 28) -> (1, 28, 1)
    features_cnn = np.expand_dims(features_scaled, axis=-1)

    # Step 5: Predict with model -> probabilities
    try:
        prediction_probs = model.predict(features_cnn)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "Error: Model prediction failed."

    # Step 6: Take argmax across classes
    predicted_index = int(np.argmax(prediction_probs))

    # Step 7: Map index to genre name
    predicted_genre = genre_mapping.get(predicted_index, "Unknown Genre")

    return predicted_genre


# ---- Streamlit UI ----
def main():
    """
    The primary function that builds and runs our Streamlit application.
    """
    st.title("🎵 Music Genre Classification App")

    st.write(
        "Welcome! This application uses a Convolutional Neural Network (CNN) "
        "to predict the genre of a music track."
    )
    st.write(
        "**Instructions:** Please upload a short audio file in `.wav` format to get started."
    )

    # Show TensorFlow warning if import failed
    if not TF_IMPORT_OK:
        st.warning(
            "⚠️ TensorFlow could not be imported. "
            "CNN-based prediction will not work until TensorFlow is fixed."
        )
        with st.expander("Show TensorFlow error details"):
            st.code(TF_ERROR_MSG, language="text")

    # --- File Uploader Widget ---
    uploaded_file = st.file_uploader(
        "Drag and drop your audio file here",
        type=["wav"],
    )

    # --- Handle uploaded file ---
    if uploaded_file is not None:
        # Success message
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # 🎧 Show audio player for the uploaded file
        st.subheader("▶️ Preview Uploaded Audio")
        st.audio(uploaded_file, format="audio/wav")

        # Agar TensorFlow available hai tabhi prediction karne ki koshish karein
        if TF_IMPORT_OK:
            # ⏳ Run prediction inside spinner
            with st.spinner("Classifying your track... 🎶"):
                predicted_genre = predict_genre(uploaded_file)

            # --- FINAL RESULT DISPLAY ---
            st.subheader("🎯 Prediction Result")
            st.markdown(f"## **{predicted_genre.capitalize()}**")
        else:
            st.info("Prediction is disabled because TensorFlow is not working right now.")


if __name__ == "__main__":
    main()
