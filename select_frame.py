import numpy as np
import cv2
import matplotlib.pyplot as plt


def select_frame(filename, out_filename):
    cap = cv2.VideoCapture(filename)

    _, frame = cap.read()
    if not _:
        return
    plt.imsave(out_filename, frame[:,:,::-1])

"""
usage: 
1. change video filename
2. change saving image filename
"""
if __name__ == '__main__':
    select_frame('data/Alita1.mp4', 'data/alita1.png')
