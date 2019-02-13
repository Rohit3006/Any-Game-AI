from selenium import webdriver
import time
import pickle
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import glob
from Deep_Neural_Net import create_deep_neural_net
from Conv_Net import create_conv_net
import pyscreenshot

df = pd.read_csv("inputs.csv")
inputs = list(set(df.Input.values))  # list(set(df.Input.values))

size_x = 70
size_y = 30
if os.path.exists("model.model"):
    model = load_model("model.model")
else:
    inputs, model = create_deep_neural_net(output_neurons=len(
        inputs), layers=4, epochs=20, size_x=size_x, size_y=size_y, train_size=0.80)

for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

frame_number = 0
paused = False

while True:
    if keyboard.is_pressed('escape'):  # break loop
        print("escape")
        break

    if keyboard.is_pressed('p'):  # break loop
        if not paused:
            print("paused")
            paused = True
        else:
            paused = False

    if paused:
        continue

    frame_number += 1
    if frame_number % 200 != 0:  # Get every 400th frame
        continue

    frame = np.array(ImageGrab.grab(bbox=(360, 220, 1080, 520)))
    #frame = np.array(pyscreenshot.grab()[479:960, 73:714])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 50, 200)
    frame = preprocessing.scale(cv2.resize(frame, (size_y, size_x)))
    frame = frame.reshape(-1, size_x*size_y)

    prediction = model.predict(frame)
    print(prediction)

    prediction = np.argmax(prediction)

    print(inputs[prediction])

    if inputs[prediction] != "blank":
        keyboard.press(inputs[prediction])
