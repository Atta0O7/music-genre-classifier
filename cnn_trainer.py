# cnn_trainer.py

# --- 1. Import necessary libraries for data handling ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 2. Import TensorFlow / Keras and required layers for CNN ---

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,             # 1D convolution layer
    MaxPooling1D,       # 1D max pooling layer
    BatchNormalization, # stabilizes training
    Dropout,            # regularization
    Flatten,            # bridge conv -> dense
    Dense               # fully-connected layers
)

# --- 3. Load and Prepare the Data ---

CSV_PATH = "features.csv"

print("Loading dataset...")

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    raise SystemExit(1)

# Ensure target column exists
if 'genre_label' not in features_df.columns:
    print("Error: 'genre_label' column not found in features.csv")
    raise SystemExit(1)

# Separate features and labels
X = features_df.drop('genre_label', axis=1)
y = features_df['genre_label']

print("\nOriginal shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Train-test split (same as before for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nAfter train-test split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# --- 4. Scale Features ---

print("\nScaling features with StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nShapes before reshaping (for CNN):")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")

# --- 5. Reshape Data for 1D CNN Input ---

print("\n--- Reshaping data for 1D CNN model ---")

# Add channel dimension at the end -> (samples, features, 1)
X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}")  # Expected: (7425, 28, 1)
print(f"X_test_cnn shape:  {X_test_cnn.shape}")   # Expected: (2475, 28, 1)

print("\ny_train and y_test shapes (targets):")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# --- 6. Build the CNN Model ---

print("\n--- Building the CNN Architecture ---")

# Initialize an empty Sequential model (canvas)
model = Sequential()
print("Sequential model canvas created successfully.")

# --- FIRST CONV–POOL–NORM BLOCK ---
model.add(Conv1D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=(X_train_cnn.shape[1], 1)  # (28, 1)
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# --- SECOND CONV–POOL–NORM BLOCK ---
model.add(Conv1D(
    filters=64,
    kernel_size=3,
    activation='relu'
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# --- THIRD CONV–POOL–NORM BLOCK ---
model.add(Conv1D(
    filters=128,
    kernel_size=3,
    activation='relu'
))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# --- FLATTEN: bridge conv base -> dense head ---
model.add(Flatten())

# --- DENSE CLASSIFICATION HEAD (hidden layer + dropout) ---
model.add(Dense(
    units=64,
    activation='relu'
))
model.add(Dropout(0.3))

# --- FINAL OUTPUT LAYER (10-class softmax) ---
model.add(Dense(
    units=10,          # 10 music genres
    activation='softmax'
))

# --- 7. Compile the Model ---

print("\n--- Compiling the CNN Model ---")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully. It is now ready to be trained.")

print("\n--- Final Model Architecture Summary ---")
model.summary()

# --- 8. Train the Model ---

print("\n--- Starting Model Training ---")

history = model.fit(
    X_train_cnn,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

print("\n--- Model Training Complete ---")

# --- 9. Plot Training History ---

print("\n--- Plotting Training and Validation History ---")

import matplotlib.pyplot as plt

def plot_history(history):
    """Plots accuracy and loss for training and validation sets."""
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # --- Accuracy plot ---
    axs[0].plot(history.history["accuracy"], label="Training Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Training and Validation Accuracy")
    axs[0].legend(loc="lower right")

    # --- Loss plot ---
    axs[1].plot(history.history["loss"], label="Training Loss")
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# Call the function to display the plots
plot_history(history)

# --- 10. Save the Trained Model ---

print("\n--- Saving the trained CNN model to disk ---")

# Saves architecture + weights + optimizer state + compile config
model.save("music_genre_cnn.h5")

print("\nModel successfully saved as 'music_genre_cnn.h5' in your project directory.")
print("You can later load it with tf.keras.models.load_model('music_genre_cnn.h5').")
