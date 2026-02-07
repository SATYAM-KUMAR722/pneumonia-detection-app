
!pip install opendatasets

import opendatasets as od
od.download("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")

import os
import random
import hashlib
import shutil
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

train_dir = "chest-xray-pneumonia/chest_xray/train"
train_data = os.listdir(train_dir)
for classes in train_data:
  class_path = os.path.join(train_dir,classes)
  img_name = os.listdir(class_path)
  random5 = random.sample(img_name,5)
  for index , image in enumerate(random5):
         image_path = os.path.join(class_path,image)
         img = Image.open(image_path)
         plt.subplot(1,5,index+1)
         plt.imshow(img)
         plt.title(classes)

for root, dirs, files in os.walk("chest-xray-pneumonia/chest_xray"):
    for file in files:
        if file.startswith("._") or "__MACOSX" in root:
            os.remove(os.path.join(root, file))

shutil.rmtree("chest-xray-pneumonia/chest_xray/__MACOSX", ignore_errors=True)
shutil.rmtree("chest-xray-pneumonia/chest_xray/chest_xray", ignore_errors=True)

print("Cleanup done ✔️")

import hashlib

def hasher_images(file_path):
    with open(file_path,'rb') as f:  # image is in binary form so use 'rb'
        img = f.read()               # read entire file as bytes
    hashed = hashlib.md5(img)        # create MD5 hash from bytes
    return hashed.hexdigest()        # return hash string


duplicate = []
images = {}
for root, dirs , files in os.walk("chest-xray-pneumonia/chest_xray"):
    for file in files:
        img_path = os.path.join(root,file)
        hash_value = hasher_images(img_path)

        if hash_value in images:
            duplicate.append((hash_value,img_path))
        else:
            images[hash_value] = img_path

len(duplicate)

# Deleting Duplicates
for hash_value,img_path in duplicate:
    duplicate_img_path = img_path

    if os.path.exists(duplicate_img_path):
        try:
            os.remove(duplicate_img_path)
        except Exception as e:
            print("Error ",e)

print("Duplicate image are removed")

import hashlib

def hasher_images(file_path):
    with open(file_path,'rb') as f:  # image is in binary form so use 'rb'
        img = f.read()               # read entire file as bytes
    hashed = hashlib.md5(img)        # create MD5 hash from bytes
    return hashed.hexdigest()        # return hash string


duplicate = []
images = {}
for root, dirs , files in os.walk("chest-xray-pneumonia/chest_xray"):
    for file in files:
        img_path = os.path.join(root,file)
        hash_value = hasher_images(img_path)

        if hash_value in images:
            duplicate.append((hash_value,img_path))
        else:
            images[hash_value] = img_path

len(duplicate)

# Number of images in training folder

train_dir = "chest-xray-pneumonia/chest_xray/train"
train_data = os.listdir(train_dir)
for classes in train_data:
    class_path = os.path.join(train_dir,classes)
    img_name = os.listdir(class_path)
    print(f"Number of images in {classes}: {len(img_name)}")

# Number of images in testing folder

test_dir = "chest-xray-pneumonia/chest_xray/test"
test_data = os.listdir(test_dir)
for classes in test_data:
    class_path = os.path.join(test_dir,classes)
    img_name = os.listdir(class_path)
    print(f"Number of images in {classes}: {len(img_name)}")

# Number of images in validation folder

val_dir = "chest-xray-pneumonia/chest_xray/val"
val_data = os.listdir(val_dir)
for classes in val_data:
    class_path = os.path.join(val_dir,classes)
    img_name = os.listdir(class_path)
    print(f"Number of images in {classes}: {len(img_name)}")

# Checking the image size

file_size = set()
for root,dirs, files in os.walk("chest-xray-pneumonia/chest_xray"):
    for file in files:
        img_path = os.path.join(root,file)

        with open(img_path,"rb") as img:
            image = Image.open(img_path)
            image_size = image.size
            file_size.add(image_size)

print(file_size)

# Checking the file size

unique_file_size = set()
for root,dirs, files in os.walk("chest-xray-pneumonia/chest_xray"):
    for file in files:
        img_path = os.path.join(root,file)
        try:
           file_size = os.path.getsize(img_path)/1024
           unique_file_size.add(file_size)
        except Exception as e:
            print("Error ",e)

print(unique_file_size)
print(sorted(unique_file_size))

import tensorflow as tf
from tensorflow.keras.models import Sequential

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "chest-xray-pneumonia/chest_xray/train",
    labels="inferred",
    label_mode="int",
    image_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "chest-xray-pneumonia/chest_xray/test",
    labels="inferred",
    label_mode="int",
    image_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "chest-xray-pneumonia/chest_xray/val",
    labels="inferred",
    label_mode="int",
    image_size=(256, 256),
    color_mode="rgb",
    batch_size=32,
    shuffle=True
)

from tensorflow.keras.layers import (
    Input, Rescaling, RandomFlip, RandomRotation,
    RandomZoom, RandomContrast)

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.15)
])

from tensorflow.keras.layers import (
    Input, Rescaling, RandomFlip, RandomRotation,
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential

base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(256, 256, 3)
)

base_model.trainable = False   # 🔒 FEATURE EXTRACTION

model = Sequential([

    Input(shape=(256, 256, 3)),

    Rescaling(1./255),
    RandomRotation(0.1),

    base_model,

    GlobalAveragePooling2D(),
    BatchNormalization(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_densenet_feature_extraction.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

model.load_weights("best_densenet_feature_extraction.keras")

# 🔥 Fine-tuning starts here

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

model.load_weights("best_densenet_feature_extraction.keras")

y_true, y_pred = [], []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype("int32")

    y_pred.extend(preds.flatten())
    y_true.extend(labels.numpy())

from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_true, y_pred) * 100)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Normal", "Pneumonia"],
    yticklabels=["Normal", "Pneumonia"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

