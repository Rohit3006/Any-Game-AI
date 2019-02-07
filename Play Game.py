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
import pickle
import pandas as pd
import glob
from Deep_Neural_Net import create_deep_neural_net
from Conv_Net import create_conv_net


if os.path.exists("model.model"):
    model = pickle.load(open("model.model", "rb"))
else:
    model = create_deep_neural_net(output_neurons=1, layers=2, epochs=3, size=50) # change line if you want conv net
    size = 50

for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

frame_number = 0

while True:
    frame_number += 1
    if frame_number % 100 != 0:  # Get every 100th frame
        continue

    frame = np.array(ImageGrab.grab(bbox=(450, 200, 950, 750)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (size,size)) / 255
    frame = frame.reshape(-1,size**2)

    print(frame)

    prediction = model.predict(frame)
    print(prediction)
    prediction = np.argmax(prediction)
    if prediction == 0:
        keyboard.press('space')
        print("Space")
    elif prediction == 1:
        print("Blank")
