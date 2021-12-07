import time
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread, imshow
import threading

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

IMG_SIZE = 48
dataset_base = "./mrlEyes_2018_01/mrlEyes_2018_01"
dataset_dir = '.'

data = []
labels = []
lock = threading.Lock()


def collect_image_in_thread(dir):
    for image_path in os.listdir(dataset_base + '/' + dir):
        # print(image_path)
        label = int(image_path.split("_")[4])
        # print(label)
        try:
            img_array = imread(dataset_base + '/' + dir + '/'+image_path)
            resized_array = resize(img_array, (IMG_SIZE, IMG_SIZE))
            # print(resized_array.shape)
            lock.acquire()
            try:
                if resized_array.shape == (IMG_SIZE, IMG_SIZE):
                    data.append(resized_array)
                    labels.append(label)

            except Exception as e2:
                print(e2)
            lock.release()
            # break
        except Exception as e:
            print(e)


def prepare_dataset():
    dirs = os.listdir(dataset_base)
    thread_list = []
    for dir in dirs:
        if dir.startswith("s0"):
            print(dir)
            thread = threading.Thread(
                target=collect_image_in_thread, args=(dir,))
            thread_list.append(thread)
            thread.start()
    for thread in thread_list:
        thread.join()
    return np.array(data), np.array(labels)


X, y = prepare_dataset()
X = X / 255

seed = 0
X_train, X2, y_train, y2 = train_test_split(
    X, y, random_state=seed, test_size=0.40)
X_val, X_test, y_val, y_test = train_test_split(
    X2, y2, random_state=seed, test_size=0.50)

X_train = np.reshape(X_train, (X_train.shape[0], IMG_SIZE, IMG_SIZE, 1))
X_val = np.reshape(X_val, (X_val.shape[0], IMG_SIZE, IMG_SIZE, 1))
X_test = np.reshape(X_test, (X_test.shape[0], IMG_SIZE, IMG_SIZE, 1))

with open(dataset_dir + '/X_train.np', mode='wb') as file:
    np.save(file, X_train)
with open(dataset_dir + '/X_val.np', mode='wb') as file:
    np.save(file, X_val)
with open(dataset_dir + '/X_test.np', mode='wb') as file:
    np.save(file, X_test)
with open(dataset_dir + '/y_train.np', mode='wb') as file:
    np.save(file, y_train)
with open(dataset_dir + '/y_val.np', mode='wb') as file:
    np.save(file, y_val)
with open(dataset_dir + '/y_test.np', mode='wb') as file:
    np.save(file, y_test)

print("Total Train data (cleased eye)", sum(y_train == 0))
print("Total Train data (open eye)", sum(y_train == 1))
print("Total Train data", len(y_train))

print("Total Validation data (cleased eye)", sum(y_val == 0))
print("Total Validation data (open eye)", sum(y_val == 1))
print("Total Validation data", len(y_val))

print("Total Test data (cleased eye)", sum(y_test == 0))
print("Total Test data (open eye)", sum(y_test == 1))
print("Total Test data", len(y_test))

print("Total data", len(y_train) + len(y_val) + len(y_test))
