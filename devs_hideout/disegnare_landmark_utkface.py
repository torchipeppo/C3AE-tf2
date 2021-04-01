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



# 1_0_2_20161219140530307.jpg -4 71 -4 96 -3 120 -1 144 9 166 28 179 53 186 77 192 100 194 121 191 142 183 161 174 180 161 192 142 195 120 194 97 192 74 16 53 29 39 48 33 68 34 86 40 113 39 129 33 148 32 164 37 175 49 100 59 101 72 101 85 101 99 78 112 89 113 100 116 110 114 120 111 39 62 51 61 61 60 71 65 60 63 50 62 124 64 134 59 144 59 155 62 144 62 134 62 55 137 72 134 87 132 97 133 107 131 120 132 136 133 121 143 109 146 98 147 88 146 72 145 61 138 87 137 97 138 107 136 130 135 108 139 98 140 88 139 
