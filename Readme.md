# Any Game AI
Uses a Convolutional Neural Network or a deep neural network to play a video game, given player input data.

# Information
The Collect_Data file is used to collect frames while you play the game. It will have a 5 second countdown before starting to do so. It will save a specified section of the screen as a png file to be used by the NN when creating the model. At the same time, user inputs are written to a csv file, as these will be the output data for the neural network. Pressing Escape stops the program.

The Deep_Neural_Net file creates the regular deep neural net and can take in various inputs, such as number of output neurons, convolutions, and epochs. The Conv_Net file creates the CNN.

The Play Game file either creates the model or loads a pre-existing one up.

model.h5 stores the model information and is what is loaded up in Play Game.
