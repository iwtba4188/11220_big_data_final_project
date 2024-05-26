import tensorflow as tf

import keras
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "..\data"
BATCH_SIZE = 32
IMG_HEIGHT = 240
IMG_WIDTH = 240
SEED = 123
num_classes = 2

model = keras.models.load_model("./model.keras")
# model.load_weights("./w.weights.h5")

# model.compile(
#     optimizer="adam",
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )

# model.load_weights(filepath="./w.weights.h5")

ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# model.predict(ds)
print(model.predict(ds))
