import numpy as np
import cv2
import time
import keyboard
import csv
from PIL import ImageGrab
import os
import keyboard
import pyscreenshot
import random

frame_number = -1

possible_inputs = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split(" ") + ["space", "ctrl", "shift"]

directory = "Frames/"
length = len([name for name in os.listdir(directory) if os.path.isfile(name)])
number = length + 1

file = open("inputs.csv", "a")
writer = csv.writer(file)

if number == 1:
	writer.writerow(["Frame Number", "Input"])

for i in range(5, 0, -1):
	print(i)
	time.sleep(1)

max_number = 5000
max_string = str(max_number)
while number < max_number:
	frame_number += 1
	if frame_number % 200 != 0: # Get every 200th frame
		continue

	frame = np.array(ImageGrab.grab(bbox=(360, 220, 1080, 520))) # Set bounds of frame to whatever for the chosen game
	#frame = np.array(pyscreenshot.grab()[479:960, 73:714])
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	string = str(number)

	if keyboard.is_pressed('escape'):
		break

	input_list = []
	for poss in possible_inputs:
		if keyboard.is_pressed(poss):
			input_list.append(poss)

	if len(input_list) == 0:
		rand_num = random.randint(0, 10) # randomly determine if blank frame saved or not
		if rand_num < 3:
			writer.writerow(["{0}".format(number), "blank"])
		else:
			continue
	else:
		input_string = "+".join(input_list)
		writer.writerow(["{0}".format(number), input_string])

	cv2.imwrite(os.path.join(directory, "Frame {0}.png".format(
		"0" * (len(max_string) - len(string)) + string)), frame)
	number += 1
