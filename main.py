import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

fp = "data\\vlc-record-2021-05-12-12h42m56s-TI_5KM_vehicle.asf-.avi"

try:
    cap = cv2.VideoCapture(fp)
except:
    cap = cv2.VideoCapture(0)

sns.set_style('darkgrid')

fig, ax = plt.subplots()

ax.set_title("Histogram")
ax.set_xlabel("Pixel Values")
ax.set_ylabel("Pixel Intensity")

bins = 256
x_data = np.arange(bins)
y_data = np.zeros((bins, 1))

plotGray, = ax.plot(x_data, y_data, c='royalblue', lw=2, label="intensity")

ax.set_xlim(0, bins-1)
ax.set_ylim(0, 0.1)
# ax.axis('auto')
# ax.set_autoscale_on(True)

ax.legend()
# plt.ion()
# plt.show()

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

    cv2.imshow("eq_norm", equalized_img)

    # plotGray.set_ydata(histogram)
    # fig.canvas.draw()
    # plt.pause(0.001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
