import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from config import BINS


def histEqContrast(img):
    num_of_pixels = img.shape[0] * img.shape[1]
    histogram = cv2.calcHist([img], [0], None, [BINS], [
                             0, BINS]) / num_of_pixels

    norm_cum_histogram = np.cumsum(histogram)

    transform_map = np.floor(255 * norm_cum_histogram).astype(np.uint8)

    img_list = list(img.flatten())
    equalized_img_lst = [transform_map[p] for p in img_list]
    equalized_img = np.reshape(np.asarray(equalized_img_lst), img.shape)

    equalized_img = cv2.bitwise_not(equalized_img)

    return equalized_img
