from sklearn.metrics import classification_report
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
model_dir = "model.h5"
dataset_dir = "."

# Retrieve Data
with open(dataset_dir + '/X_test.np', mode='rb') as file:
    X_test = np.load(file)
with open(dataset_dir + '/y_test.np', mode='rb') as file:
    y_test = np.load(file)

print("Total Test data (cleased eye)", sum(y_test == 0))
print("Total Test data (open eye)", sum(y_test == 1))
print("Total Test data", len(y_test))

# Retrieving model
model = tf.keras.models.load_model(model_dir)

prediction = model.predict(X_test)
prediction = (prediction > 0) * 1

labels = ["Closed", "Open"]

print(classification_report(y_test, prediction, target_names=labels))

model.evaluate(x=X_test, y=y_test)
