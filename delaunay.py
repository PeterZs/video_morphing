import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2


def delaunay(points):
    tri = Delaunay(points)
    return tri


if __name__ == '__main__':
    img1 = cv2.imread('data/alita1.png')
    img2 = cv2.imread('data/alita2.png')

    # load corresponding points from .mat file
    #mat = scipy.io.loadmat('points.mat')
    #p1 = mat['p1']
    #p2 = mat['p2']

    # load corresponding points from .npy file
    p1 = np.load('data/alita_frame1_pts1.npy')
    p2 = np.load('data/alita_frame1_pts2.npy')
    
    p1 = np.transpose(p1)
    p2 = np.transpose(p2)

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    p1 = np.concatenate((p1, np.array([[0, 0], [0, h1-1], [w1-1,0], [w1-1, h1-1]])), axis=0)
    p2 = np.concatenate((p2, np.array([[0, 0], [0, h2-1], [w2-1,0], [w2-1, h2-1]])), axis=0)
    print(p1.shape)

    tri = delaunay(p1)

    tri_points = p1[tri.simplices]
    tri_points2 = p2[tri.simplices]

    for p in p1:
        cv2.circle(img1, tuple(p), 2, (255, 0, 0), -1)
    for p in p2:
        cv2.circle(img2, tuple(p), 2, (255, 0, 0), -1)
    
    for t in tri_points:
        cv2.line(img1, tuple(t[0]), tuple(t[1]), (255, 0, 0), 1)
        cv2.line(img1, tuple(t[1]), tuple(t[2]), (255, 0, 0), 1)
        cv2.line(img1, tuple(t[0]), tuple(t[2]), (255, 0, 0), 1)
    for t in tri_points2:
        cv2.line(img2, tuple(t[0]), tuple(t[1]), (255, 0, 0), 1)
        cv2.line(img2, tuple(t[1]), tuple(t[2]), (255, 0, 0), 1)
        cv2.line(img2, tuple(t[0]), tuple(t[2]), (255, 0, 0), 1)

    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.imshow('image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('data/alita1_delaunay.png', img1)
    cv2.imwrite('data/alita2_delaunay2.png', img2)
