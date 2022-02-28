import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# fp = "data\\vlc-record-2021-05-12-12h42m56s-TI_5KM_vehicle.asf-.avi"
fp = "imgs\\img-2.jfif"
try:
    cap = cv2.VideoCapture(fp)
except:
    cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)

sns.set_style('darkgrid')

bins = 256

while(cap):
    ret, frame = cap.read()
    if not ret:
        break

    num_of_pixels = frame.shape[0] * frame.shape[1]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", frame)

    histogram = cv2.calcHist([frame], [0], None, [bins], [
                             0, bins]) / num_of_pixels

    norm_cum_histogram = np.cumsum(histogram)

    transform_map = np.floor(255 * norm_cum_histogram).astype(np.uint8)

    img_list = list(frame.flatten())
    equalized_img_lst = [transform_map[p] for p in img_list]
    equalized_img = np.reshape(np.asarray(equalized_img_lst), frame.shape)

    equalized_img = cv2.bitwise_not(equalized_img)

    cv2.imshow("eq_norm", equalized_img, )

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
