from cProfile import label
from matplotlib import markers
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

fp = "imgs\\img-1.jpg"

img = cv2.imread(fp)

sns.set_style('darkgrid')

fig, ax = plt.subplots()

ax.set_title("Histogram")
ax.set_xlabel("Pixel Values")
ax.set_ylabel("Pixel Intensity")

bins = 256

ax.set_xlim(0, bins-1)
ax.set_ylim(0, 0.10)

ax.legend()

num_of_pixels = img.shape[0] * img.shape[1]

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)

histogram = cv2.calcHist([img], [0], None, [bins], [
    0, 255]) / num_of_pixels

plt.plot(histogram)
plt.show()
