import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import BINS
from hist_eq_contrast import histEqContrast
from edge_conv import sharpenEdges
# from config import out_path


def write_video(frames_list, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'result_equ_img_{np.random.randint(0, 99)}.mp4', fourcc, fps, (
        frames_list[0].shape[1], frames_list[0].shape[0]), False)
    for i in range(len(frames_list)):
        out.write(frames_list[i])
    print("Saved")
    out.release()


fp = "data\\vlc-record-2021-05-12-12h42m56s-TI_5KM_vehicle.asf-.avi"
# fp = "data\\imgs\\img-2.jfif"

equalized_img_frames = []
input_frames = []

try:
    cap = cv2.VideoCapture(fp)
except:
    cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)

while(cap):
    ret, frame = cap.read()
    if not ret:
        write_video(equalized_img_frames, fps)
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # input_frames.append(frame)
    cv2.imshow("frame", frame)

    equalized_img = histEqContrast(frame)
    equalized_img_edges_conv = sharpenEdges(equalized_img)

    equalized_img_frames.append(equalized_img_edges_conv)

    # concatenated_vid = [np.hstack((input_frames[i], np.zeros(
    #     (input_frames[0].shape[0], 10)), equalized_img_frames[i])).astype(np.float32) for i in range(len(input_frames))]

    cv2.imshow("eq_norm_sharpened", equalized_img_edges_conv)

    if cv2.waitKey(int(fps)) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        write_video(equalized_img_frames, fps)
        break
