import cv2
import numpy as np


def sharpenEdges(img):
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)

    diff = cv2.subtract(img, dst)
    final = cv2.add(diff, img)
    return final
