import numpy as np
import cv2
import time
import keyboard
import csv
from PIL import ImageGrab
import os
import keyboard

frame_number = -1
number = 1

directory = "Frames/"
length = len([name for name in os.listdir(directory) if os.path.isfile(name)])
if length > 0:
    number = length # Continue progress

file = open("inputs.csv", "w")
writer = csv.writer(file)
writer.writerow(["Frame Number", "Input"])

for i in range(5, 0, -1):
    print(i)
    time.sleep(1)

while True:
    frame_number += 1
    if frame_number % 500 != 0: # Get every 500th frame
        continue

    frame = np.array(ImageGrab.grab(bbox=(450, 200, 950, 750))) # Set bounds of frame to whatever for the chosen game
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if keyboard.is_pressed('escape'):
        break
    elif keyboard.is_pressed('space'):
        writer.writerow(["{0}".format(number), "space"])
    else:
        writer.writerow(["{0}".format(number), "blank"])

    cv2.imwrite(os.path.join(directory, "Frame {0}.png".format(number)), frame)
    number += 1
