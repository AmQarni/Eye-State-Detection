# CSCI 7850
## Project: Driver Drowsiness Detection
-----

## Purpose:
Road accidents can take human life. One of the important causes of car accidents is the drowsiness of the driver. This project tries to minimize road accidents by utilizing the driver’s eye state (Closed or Open) to detect the driver’s drowsiness. In this project, we propose a method by which we can alert them when they feel drowsy. We use the Convolutional Neural Network (CNN) to build our model. 


## TL;DR:
To deploy/use the project you need to build up python environment first. I have used popular packages like numpy, pandas, sklearn, skimage, and tensorflow library or framework in this project. You need to make sure that all the packages are installed. Follow the following steps to use the project.

First clone this project by giving the using the following command in terminal or command prompt.

```
git clone https://github.com/AmQarni/Eye-State-Detection.git
```

### Running the pretrained model
Use the following command to run our pretrained model.

```
python3 model.py
```
-----

## Building our DL model 
For building our model from scratch you need to follow the following steps.

1. Go to this Google drive [link](https://drive.google.com/drive/folders/1I6t3FNLm8uSehRAcN6KbM3d-vpgRFnrQ?usp=sharing) and download all the files and folders and place in the same directory where you have cloned this project.
2. Follow the following discussion-

### Dataset Creation
For creating the dataset you need to make sure that the mrlEyes_2018_01 directory and the preprocess-data.py file are in the same directory. You can also resolve the path in preprocess-data.py file also.
Use the following command to run the preprocessing step.

```
python3 preprocess-data.py
```

### Training the model
For training make sure you have the train-model.py, X_train.np, y_train.np, X_val.np, and y_val.np are under the same directory. Then use the following command to run the training.

```
python3 train-model.py
```

### Testing the model
For testing make sure you have the test-model.py, X_test.np, and y_test.np are under the same directory. Then use the following command to run the testing.

```
python3 test-model.py
```

-----

## Advantages:
The main advantage of this project is we can make prediction of eye state under very small amount of time (26 ms using NVIDIA GeForce RTX 2080 GPU) and our model takes only 1.2 MBs of storage. So, it is both memory efficient and fast.

## Shortcomings:
This project only detects the eye state of the an eye. It can be used directly. We need to implement the other parts like capturing driver images and extracting eyes from the images to make it practically feasible for using in practical situation. 

## Contributors:
Amjad Alqarni

## Copyright 2021 - MTSU CS

