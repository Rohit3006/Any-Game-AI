# CNN Game AI
Uses a Convolutional Neural Network to play a video game, given player input data (Have yet to train and test model; is still in development stage)

# Information
The Collect_Data file is used to collect frames while you play the game. It will have a 5 second countdown before starting to do so. It will save a specified section of the screen as a png file to be used by the CNN when creating the model. At the same time, user inputs are written to a csv file, as these will be the output data for the neural network. Pressing Escape stops the program.

The Create_Neural_Net file creates the CNN and can take in various inputs, such as number of output neurons, convolutions, and epochs

The Play Game file opens up a browser using selenium and plays the game based on the model it's built.