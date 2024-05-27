from keras.applications import VGG16
from keras import models
from keras.layers import Dense, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(tf.test.is_built_with_cuda())
# input()

# Load VGG16 model without top layers
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=[224, 224, 3])

# freeze pretrained layer
for layer in vgg16.layers:
    layer.trainable = False

# build fully connected layers
x = Flatten()(vgg16.output)
x = Dense(units=256, activation="relu")(x)
x = Dense(units=2, activation="softmax")(x)

# create model
model = models.Model(inputs=vgg16.input, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# prepare
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
train_df = datagen.flow_from_directory(
    "../data",
    batch_size=32,
    class_mode="categorical",
    target_size=(224, 224),
    subset="training",
)
validation_df = datagen.flow_from_directory(
    "../data",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)


# train model
history = model.fit(train_df, epochs=3, validation_data=validation_df)
model.save("./vgg16_model.keras", overwrite=True)

# linear plot
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()
