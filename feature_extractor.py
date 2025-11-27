import os
import librosa
import numpy as np
import pandas as pd   # pandas import

# -----------------------------
# PATH SETTINGS
# -----------------------------
DATASET_PATH = "genres_original"
CSV_PATH = "features.csv"   # final output CSV

# -----------------------------
# AUDIO SETTINGS
# -----------------------------
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10

# MFCC / STFT SETTINGS
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS


def process_dataset(dataset_path, csv_path):

    data = {
        "mapping": [],
        "labels": [],
        "features": []
    }

    print("Starting feature extraction...")

    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path, genre_folder)

        if os.path.isdir(genre_path):
            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")

            for filename in sorted(os.listdir(genre_path)):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)

                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                        if len(signal) >= SAMPLES_PER_TRACK:
                            num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

                            for s in range(NUM_SEGMENTS):
                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment
                                segment = signal[start_sample:end_sample]

                                # Extract features
                                mfccs_processed = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH), axis=1)
                                chroma_processed = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH), axis=1)
                                spectral_centroid_processed = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH))
                                spectral_rolloff_processed = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH))
                                zcr_processed = np.mean(librosa.feature.zero_crossing_rate(y=segment, hop_length=HOP_LENGTH))

                                # Combine features into a single vector
                                feature_vector = np.hstack((mfccs_processed, chroma_processed, spectral_centroid_processed, spectral_rolloff_processed, zcr_processed))

                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i)

                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue

    # Convert to DataFrame
    print("\nConverting data to pandas DataFrame...")
    features_df = pd.DataFrame(data["features"])
    features_df["genre_label"] = data["labels"]

    # Save DataFrame to CSV
    print(f"Saving DataFrame to {csv_path}...")
    features_df.to_csv(csv_path, index=False)

    print("\nFeature extraction complete. The file 'features.csv' has been created.")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, CSV_PATH)
