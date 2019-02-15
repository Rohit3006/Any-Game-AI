from selenium import webdriver
import time
import pickle
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import glob
import keras
from Networks.Deep_Neural_Net import create_deep_neural_net, train
from Networks.Conv_Net import create_conv_net

MODEL_TYPE = "deep" # MODEL_TYPE should be either deep or conv

df = pd.read_csv("inputs.csv")
inputs = list(set(df.Input.values))

size_x = 70
size_y = 30
if os.path.exists("model.h5"):
    model = keras.models.load_model('model.h5')
else:
    if MODEL_TYPE == "deep":
        model = create_deep_neural_net(output_neurons=len(inputs), layers=5)
        inputs, model = train(model, epochs=30, size_x=size_x,size_y=size_y, train_size=0.8)

    elif MODEL_TYPE == "conv":
        inputs, model = create_conv_net(output_neurons=len(inputs), layers=2, epochs=15, size_x=size_x, size_y=size_y,train_size=0.75)

for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

frame_number = 0
paused = False

while True:
    if keyboard.is_pressed('escape'):  # break loop
        print("escape")
        break

    if keyboard.is_pressed('p'):  # pause
        if not paused:
            print("paused")
            paused = True
        else:
            paused = False

    if paused:
        continue

    frame_number += 1
    if frame_number % 200 != 0:  # Get every 200th frame
        continue

    frame = np.array(ImageGrab.grab(bbox=(360, 220, 1080, 520)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if MODEL_TYPE == "conv":
        frame = cv2.resize(frame, (size_y, size_x))
        frame = frame / 255
        frame = tf.Session().run(tf.expand_dims(frame, 0))
    elif MODEL_TYPE == "deep":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, 100, 200)
        frame = preprocessing.scale(cv2.resize(frame, (size_y, size_x)))
        frame = frame.reshape(-1, size_x*size_y)

    prediction = model.predict(frame)
    print(prediction)

    prediction = np.argmax(prediction)
    key_press = inputs[prediction]

    print(key_press)

    if key_press != "blank":
        keyboard.press(key_press)
