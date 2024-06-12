# Code ref:
import tensorflow as tf

import keras
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "..\data"
BATCH_SIZE = 32
IMG_HEIGHT = 240
IMG_WIDTH = 240
SEED = 123

keras.utils.set_random_seed(SEED)

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

validation_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)


class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# train_ds.take(1)[0][0]

# 如果是用 jupyter notebook 不用加這行就會顯示
plt.show(block=True)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# input()
# Normalization Layer
# normalization_layer = keras.layers.Rescaling(1.0 / 255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# input()

num_classes = 2

model = keras.Sequential(
    [
        # keras.layers.InputLayer(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.save("model.keras", overwrite=True)

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=validation_ds, epochs=3)
# model.save_weights("./w.weights.h5", overwrite=True)


plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

model

test_loss, test_acc = model.evaluate(validation_ds, verbose=2)
print(test_acc)

input()
