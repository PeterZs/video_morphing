import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import time

def combine_video(box=None):
    cap1 = cv2.VideoCapture('data/Alita1.mp4')
    cap2 = cv2.VideoCapture('data/Alita2.mp4')
    out = cv2.VideoWriter('data/blending.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (960, 800))

    fac = 1.0/(40-1)
    cnt = 0
    for alpha in np.arange(0, 1, fac):
        #frame = np.zeros((1440, 1280, 3))
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        if cnt == 0:
            frame = frame1
            frame = np.uint8(frame)
        else:
            frame = (1-alpha)*frame1 + alpha*frame2
            frame = np.uint8(frame)

        out.write(frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(33)
        if key == 33:
            break
        
        cnt += 1
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    out.write(frame2)

    out.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    combine_video()
