import tensorflow as tf
from keras import layers, models
from keras.utils import image_dataset_from_directory
import os

# Dataset settings
dataset_dir = "Indian"   # EXACT folder name from explorer
img_size = (64, 64)
batch_size = 32
epochs = 10

# Load dataset (Keras 3 way)
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

num_classes = train_ds.element_spec[1].shape[-1]

# Normalize images
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save model
model.save("sign_language_interpreter_model.keras")
