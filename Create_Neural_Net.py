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

def create_neural_net(output_neurons = 1, layers = 1, epochs = 10, verbose = 2):
    file_dir = "Frames/"
    x = []

    files = sorted(os.listdir(file_dir))

    for file in files:
        file_name = os.path.join(file_dir, file)
        array = cv2.imread(file_name)
        x.append(array)

    x = np.array(x) / 255

    df = pd.read_csv("inputs.csv")
    y = np.array(df["Input"].apply(lambda x: replace(df, "Input", x)))

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = Sequential()

    for i in range(layers):
        model.add(Conv2D(100, (2,2)))
        model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(output_neurons, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=2, validation_data = (x_test, y_test))

    pickle.dump(model, open("model.model"))

    return model