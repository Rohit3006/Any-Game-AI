import time
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard
import os
from sklearn import preprocessing
import pandas as pd
from keras.models import load_model
from Networks.Deep_Neural_Net import create_deep_neural_net, train
from Networks.Conv_Net import create_conv_net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_TYPE = "deep"  # MODEL_TYPE should be either deep or conv

df = pd.read_csv("inputs.csv")
inputs = list(set(df.Input.values))

size_x = 100
size_y = 70

ROI = [100, 225, 50, 500]
if os.path.exists("model2.h5"):
    model = load_model('model.h5')
else:
    if MODEL_TYPE == "deep":
        model = create_deep_neural_net(output_neurons=len(inputs), layers=5)
        inputs, model = train(model, epochs=10, size_x=size_x, size_y=size_y, train_size=0.8, roi=ROI)

    elif MODEL_TYPE == "conv":
        inputs, model = create_conv_net(output_neurons=len(inputs), layers=1, epochs=5, size_x=size_x, size_y=size_y,train_size=0.75)

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
        frame = frame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, 100, 200)
        prediction = model.predict(frame)
    elif MODEL_TYPE == "deep":
        frame = frame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, 100, 200)

        cv2.imwrite("img.png", frame)

        frame = preprocessing.scale(cv2.resize(frame, (size_x, size_y)))
        frame = frame.reshape(-1, size_x*size_y)
        prediction = model.predict(frame)

    print(prediction)

    prediction = np.argmax(prediction)
    key_press = inputs[prediction]

    print(key_press)

    if key_press != "blank":
        keyboard.press(key_press)
