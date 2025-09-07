import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Example model (change this to your actual architecture)
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(26, activation="softmax")  # Assuming 26 classes (A-Z)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Build model
model = build_model()

# (Optional) Dummy training if you don’t have dataset plugged in yet
import numpy as np
x_dummy = np.random.rand(100, 64, 64, 1)  # 100 random grayscale images
y_dummy = keras.utils.to_categorical(np.random.randint(26, size=100), 26)
model.fit(x_dummy, y_dummy, epochs=1)

# ✅ Save in new recommended format
model.save("models/sign_model.keras")

# ✅ Save in legacy HDF5 format
model.save("models/sign_model.h5", save_format="h5")

print("✅ Model saved as sign_model.keras and sign_model.h5")
