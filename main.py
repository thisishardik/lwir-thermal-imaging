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

while(cap):
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
