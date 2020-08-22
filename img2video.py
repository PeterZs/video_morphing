import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import time

def img2video(box=None):
    out = cv2.VideoWriter('data/Alita_flow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (960, 800))

    for i in range(1, 40):
        frame = cv2.imread('data/experiment/alita_step40/of_step40/frame_%d.png' % i)

        out.write(frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(33)
        if key == 33:
            break

    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img2video()
