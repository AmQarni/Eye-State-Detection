import time
import numpy as np
import pandas as pd
import os
#import cv2
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.io import imread, imshow

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

model_dir = "model.h5"
image_path = "./images/1.png"

labels = ["Closed", "Open"]
IMG_SIZE = 48

model = tf.keras.models.load_model(model_dir)
model.summary()


def prepare(filepath):
    img_array = imread(filepath)
    # print(img_array.shape)
    resized_array = resize(img_array, (1, IMG_SIZE, IMG_SIZE, 1))
    resized_array = resized_array / 255
    # print(resized_array.shape)
    return resized_array


def predict(filepath):
    prediction = model.predict(prepare(filepath))
    return labels[((prediction > 1)*0)[0][0]]

imshow(image_path)
# imshow(prepare(image_path))

start = time.time()
prediction = predict(image_path)
end = time.time()

print("Prediction", prediction)
print("Time", end-start)


