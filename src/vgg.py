import keras


# Load keras.applications.VGG16 model without top layers
vgg16 = keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=[224, 224, 3]
)
# freeze pretrained layer
for layer in vgg16.layers:
    layer.trainable = False
# build fully connected layers
x = keras.layers.Flatten()(vgg16.output)
x = keras.layers.Dense(units=256, activation="relu")(x)
x = keras.layers.Dense(units=2, activation="softmax")(x)

# create model
model = keras.Model(inputs=vgg16.input, outputs=x)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# prepare

train_df = keras.utils.image_dataset_from_directory(
    "../data",
    batch_size=32,
    target_size=(224, 224),
    validation_split=0.2,
    seed=123,
    subset="training",
)
validation_df = keras.utils.image_dataset_from_directory(
    "../data",
    target_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    seed=123,
    subset="validation",
)

normalization_layer = keras.layers.Rescaling(1.0 / 255)
train_df = train_df.map(lambda x, y: (normalization_layer(x), y))
validation_df = validation_df.map(lambda x, y: (normalization_layer(x), y))

# train model

history = model.fit(
    train_df, steps_per_epoch=train_df.samples // train_df.batch_size, epochs=3
)
# evluate
try:
    loss, accuracy = model.evaluate(
        validation_df, steps=validation_df.samples // validation_df.batch_size
    )
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
except Exception as e:
    print(f"Error during evaluation: {e}")
