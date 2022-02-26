from cProfile import label
from matplotlib import markers
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

fp = "data\\TI_vehicle 8.7km.asf"

try:
    cap = cv2.VideoCapture(fp)
except:
    cap = cv2.VideoCapture(0)

sns.set_style('darkgrid')

fig, ax = plt.subplots()

ax.set_title("Histogram")
ax.set_xlabel("Pixel Values")
ax.set_ylabel("Pixel Intensity")

bins = 11
x_data = np.arange(bins)
y_data = np.zeros((bins, 1))

plotGray, = ax.plot(x_data, y_data, c='r', lw=2, label="intensity")

ax.set_xlim(0, bins-1)
ax.set_ylim(0, 1)


ax.legend()
plt.ion()
plt.show()

while(cap):
    ret, frame = cap.read()
    if not ret:
        break

    num_of_pixels = frame.shape[0] * frame.shape[1]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", frame)

    histogram = cv2.calcHist([frame], [0], None, [bins], [
                             0, 255]) / num_of_pixels

    plotGray.set_ydata(histogram)
    fig.canvas.draw()
    plt.pause(0.001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
