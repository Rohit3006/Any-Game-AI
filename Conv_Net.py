from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import pickle
import pandas as pd
import glob

def replace(df, column, x):
    unique = list(set(df[column].values))
    return unique.index(x)

def create_conv_net(output_neurons=1, layers=1, epochs=10, verbose=2, size=20):
    file_dir = "Frames/"
    x = []

    files = sorted(os.listdir(file_dir))

    for file in files:
        file_name = os.path.join(file_dir, file)
        array = cv2.imread(file_name)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        array = cv2.resize(array, (size, size))
        x.append(array)

    x = np.array(x) / 255

    df = pd.read_csv("inputs.csv")
    y = np.array(df["Input"].apply(lambda x: replace(df, "Input", x)))

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9)
    #x_train = x_train.reshape(-1, size**2)
    #x_test = x_test.reshape(-1, size**2)
    model = Sequential()

    model.add(Dense(size**2))
    for i in range(layers):
        model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_neurons, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=2, validation_data=(x_test, y_test))

    return model
