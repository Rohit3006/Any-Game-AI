import os

for file in os.listdir("Frames/"):
    os.remove(file)

os.remove("inputs.csv")
print("Cleared all frames")