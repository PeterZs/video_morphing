import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import time

def combine_video(box=None):
    cap1 = cv2.VideoCapture('data/blending.mp4')
    cap2 = cv2.VideoCapture('data/experiment/alita_step40_sametri/alita_result.mp4')
    out = cv2.VideoWriter('data/compare_blending.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 800))

    while True:
        frame = np.zeros((800, 1920, 3))
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        frame[:, :960, :] = frame1[:, :, :]
        frame[:, 960:, :] = frame2[:, :, :]
        frame = np.uint8(frame)

        out.write(frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(33)
        if key == 33:
            break

    out.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    combine_video()
