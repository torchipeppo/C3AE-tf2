import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

folder_of_interest=r"C:\Users\Giovanni\Universita\M_Anno_I\Neural Networks\dataset\UTKFace\in_the_wild"

with open(os.path.join(folder_of_interest, "landmark_list.txt"), "r") as landmark_file:
    line = landmark_file.readline()

line = line.split(" ")
print(line)
fname = line[0]
landmarks = line[1:-1]
landmarks = list(map(int,landmarks))
landmarks = np.array(landmarks)
landmarks = np.reshape(landmarks, (-1,2))
lx = landmarks[:,0]
ly = landmarks[:,1]

print(landmarks)

image = Image.open(os.path.join(folder_of_interest, "images", fname))

plt.figure()
#plt.imshow(image)
plt.scatter(lx, ly)
plt.show()
