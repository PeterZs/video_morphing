import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import time

def img2video(box=None):
    cap = cv2.VideoCapture('data/compare_sugar.mp4')
    outfolder = 'data/compare_sugar/'

    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(outfolder+'%d.png'%cnt, frame)
        
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(33)
        if key == 33:
            break
        
        cnt += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    img2video()
