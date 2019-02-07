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

possible_inputs = "abcdefghijklmnopqrstuvwxyz".split() + ["ctrl"]

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

max_number = 5000
max_string = str(max_number)
while number < max_number:
    frame_number += 1
    if frame_number % 100 != 0: # Get every 100th frame
        continue

    frame = np.array(ImageGrab.grab(bbox=(450, 200, 950, 750))) # Set bounds of frame to whatever for the chosen game
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    string = str(number)

    if keyboard.is_pressed('escape'):
        break

    input_list = []
    for poss in possible_inputs:
        if keyboard.is_pressed(poss):
            input_list.append(poss)

    if len(input_list) == 0:
        writer.writerow(["{0}".format(number), "blank"])
    else:
        input_string = "+".join(input_list)
        writer.writerow(["{0}".format(number), input_string])

    cv2.imwrite(os.path.join(directory, "Frame {0}.png".format(
        "0" * (len(max_string) - len(string)) + string)), frame)
    number += 1
