import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2
from delaunay import delaunay
import time


def create_tri_mask(sz, pts):
    mask = np.zeros(sz)
    mask = cv2.fillConvexPoly(mask, pts, 1.0, 16, 0)
    return mask


def morphing_frame(img1, img2, pts1, pts2, tri_simplices, alpha):
    """
    Args:
        img1 - numpy array with shape (H, W, 3)
        img2 - numpy array with shape (H, W, 3)
        pts1 - points for image1, np array with shape (N, 2)
        pts2 - points for image2, np array with shape (N, 2)
        tri_simplices - tri_simplices, see scipy.spatial.Delaunay
        alpha - float
    Return:
        final_res - np.uint8, (H, W, 3), morphing image
    """
    h, w, c = img1.shape

    # get middle points
    mid_pts = np.int32(alpha*pts1 + (1-alpha)*pts2)

    p1_tri = pts1[tri_simplices]
    p2_tri = pts2[tri_simplices]
    mid_tri = mid_pts[tri_simplices]

    final_res = np.zeros(img1.shape, np.uint8)
    for i in range(len(mid_tri)):
        transform_mat = cv2.getAffineTransform(np.float32(p1_tri[i]), np.float32(mid_tri[i]))
        res = cv2.warpAffine(img1, transform_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        transform_mat2 = cv2.getAffineTransform(np.float32(p2_tri[i]), np.float32(mid_tri[i]))
        res2 = cv2.warpAffine(img2, transform_mat2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        res = alpha*res + (1-alpha)*res2

        mask = create_tri_mask((h, w), mid_tri[i])
        mask = np.stack((mask,mask,mask), axis=-1)
        final_res = np.add(final_res*(1-mask), res*mask)
        
        #cv2.imshow('image', np.uint8(final_res))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    #cv2.imshow('image', np.uint8(final_res))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return np.uint8(final_res)


if __name__ == '__main__':
    img1 = cv2.imread('data/alita1.png')
    img2 = cv2.imread('data/alita2.png')

    # load corresponding points from .npy file
    p1 = np.load('data/alita_frame1_pts1.npy')
    p2 = np.load('data/alita_frame1_pts2.npy')
    
    p1 = np.transpose(p1)
    p2 = np.transpose(p2)

    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    p1 = np.concatenate((p1, np.array([[0, 0], [0, h1-1], [w1-1,0], [w1-1, h1-1]])), axis=0)
    p2 = np.concatenate((p2, np.array([[0, 0], [0, h2-1], [w2-1,0], [w2-1, h2-1]])), axis=0)

    tri = delaunay(p1)
    tri_points = p1[tri.simplices]
    tri_points2 = p2[tri.simplices]

    cnt = 0
    for alpha in np.arange(0, 1, 0.05):
        mid_frame = morphing_frame(img1, img2, p1, p2, tri.simplices, 1-alpha)
        cv2.imwrite('data/result_alita/frame_%d.png'%cnt, mid_frame)
        cnt += 1
        
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

    '''
    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.imshow('image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('data/alita1_delaunay.png', img1)
    cv2.imwrite('data/alita2_delaunay2.png', img2)
    '''