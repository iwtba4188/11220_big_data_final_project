# import tensorflow as tf

# from keras.src.legacy.preprocessing.image import ImageDataGenerator
# import keras
# import matplotlib.pyplot as plt
# import numpy as np

# import shutil
# import os

# DATA_DIR = "..\data"
# BATCH_SIZE = 32
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# SEED = 123
# num_classes = 2


# keras.utils.set_random_seed(SEED)

# # model = keras.models.load_model("./model.keras")
# model = keras.models.load_model("./vgg16_model.keras")
# # model.load_weights("./w.weights.h5")

# # model.compile(
# #     optimizer="adam",
# #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# #     metrics=["accuracy"],
# # )

# # model.load_weights(filepath="./w.weights.h5")
# datagen = ImageDataGenerator(rescale=1.0 / 255)
# ds = datagen.flow_from_directory(
#     "../data",
#     batch_size=1,
#     class_mode="categorical",
#     target_size=(224, 224),
#     shuffle=False,
# )
# # ds = keras.utils.image_dataset_from_directory(
# #     DATA_DIR,
# #     image_size=(IMG_HEIGHT, IMG_WIDTH),
# #     batch_size=BATCH_SIZE,
# # )

# # l = []
# # for image, label in ds:
# #     print(list(label.numpy()))
# #     l.extend(list(label.numpy()))

# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )

# true_labels = ds.classes
# file_names = ds.filenames
# # 預測
# predictions = model.predict(ds)
# predicted_labels = np.argmax(predictions, axis=1)

# # 找出預測錯誤的圖片
# incorrect_indices = np.where(predicted_labels != true_labels)[0]

# # 確保錯誤資料夾存在
# error_path = "./wrong"
# if not os.path.exists(error_path):
#     os.makedirs(error_path)

# # 複製錯誤圖片到錯誤資料夾
# for i in incorrect_indices:
#     src_file = os.path.join(DATA_DIR, file_names[i])
#     dst_file = os.path.join(error_path, os.path.basename(file_names[i]))
#     shutil.copy(src_file, dst_file)
# # a = model.predict_on_batch(ds)
# # print(y_classes)

# generate by ChatGPT
import os
import shutil
import numpy as np
from keras.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# 定義路徑
model_path = "./vgg16_model.keras"
data_path = "../data"
error_path = "./predict_fail"

# 載入模型
model = load_model(model_path)

# 資料預處理
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
data_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)

# 獲取實際標籤
true_labels = data_generator.classes

# 獲取檔案名稱
file_names = data_generator.filenames

# 預測
predictions = model.predict(data_generator)
predicted_labels = np.argmax(predictions, axis=1)

# 找出預測錯誤的圖片
incorrect_indices = np.where(predicted_labels != true_labels)[0]

# 確保錯誤資料夾存在
if not os.path.exists(error_path):
    os.makedirs(error_path)

# 複製錯誤圖片到錯誤資料夾
for i in incorrect_indices:
    src_file = os.path.join(data_path, file_names[i])
    dst_file = os.path.join(error_path, os.path.basename(file_names[i]))
    shutil.copy(src_file, dst_file)

print(f"共複製了 {len(incorrect_indices)} 張預測錯誤的圖片到 {error_path}。")
