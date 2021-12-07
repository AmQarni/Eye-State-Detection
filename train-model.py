import time
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

IMG_SIZE = 48
dataset_base = "./mrlEyes_2018_01/mrlEyes_2018_01"
dataset_dir = '.'

# Retrieve Data

with open(dataset_dir + '/X_train.np', mode='rb') as file:
    X_train = np.load(file)
with open(dataset_dir + '/X_val.np', mode='rb') as file:
    X_val = np.load(file)
with open(dataset_dir + '/y_train.np', mode='rb') as file:
    y_train = np.load(file)
with open(dataset_dir + '/y_val.np', mode='rb') as file:
    y_val = np.load(file)


print("Total Train data (cleased eye)", sum(y_train == 0))
print("Total Train data (open eye)", sum(y_train == 1))
print("Total Train data", len(y_train))

print("Total Validation data (cleased eye)", sum(y_val == 0))
print("Total Validation data (open eye)", sum(y_val == 1))
print("Total Validation data", len(y_val))


# Model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(3, 3))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(3, 3))

model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="tanh"))

model.compile(loss="binary_crossentropy",
              metrics=["accuracy"], optimizer="adam")

model.summary()

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=64)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()

# Saving model
model.save("model.h5")
