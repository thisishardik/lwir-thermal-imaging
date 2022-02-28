import cv2
import numpy as np

fp = "data\\TI_vehicle 8.7km.asf"

try:
    cap = cv2.VideoCapture(fp)
except:
    cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)

    diff = cv2.subtract(img, dst)
    final = cv2.add(diff, img)
    cv2.imshow('inp', img)
    cv2.imshow('final', final)
    if cv2.waitKey(int(fps)) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
