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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
import pandas as pd
import glob
from Deep_Neural_Net import create_deep_neural_net
from Conv_Net import create_conv_net

df = pd.read_csv("inputs.csv")
inputs = list(set(df.Input.values)) #list(set(df.Input.values))

size = 50
if os.path.exists("model.model"):
    model = pickle.load(open("model.model", "rb"))
else:
    inputs, model = create_deep_neural_net(output_neurons=len(inputs), layers=4, epochs=10, size=size, train_size=0.7)

for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

frame_number = 0

while True:
    if keyboard.is_pressed('Q'): # break loop
        print("escape")
        break

    frame_number += 1
    if frame_number % 100 != 0:  # Get every 100th frame
        continue

    frame = np.array(ImageGrab.grab(bbox=(450, 200, 950, 750)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 0, 200)
    frame = preprocessing.scale(cv2.resize(frame, (size, size)))
    frame = frame.reshape(-1,size**2)

    prediction = model.predict(frame)
    print(prediction)

    prediction = np.argmax(prediction)

    print(inputs[prediction])

    if inputs[prediction] != "blank":
        keyboard.press(inputs[prediction])
