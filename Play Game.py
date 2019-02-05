from selenium import webdriver
import time
import pickle
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard
import os
from Create_Neural_Net import create_neural_net

if os.path.exists("model.model"):
    model = pickle.load(open("model.model", "rb"))
else:
    model = create_neural_net(output_neurons=1, layers=1)

driver = webdriver.Firefox()
driver.get("https://flappybird.io/") # Change this to whatever website

time.sleep(5)

while True:
    # Set bounds of frame to whatever for the chosen game
    frame = ImageGrab.grab(bbox=(450, 200, 950, 750))
    prediction = np.argmax(model.predict(frame))
    if prediction == 0:
        keyboard.press('space')
