from PIL import Image
import matlab.engine
import numpy as np


def pil_img_to_list(img):
    img = list(img.getdata())
    img = [list(x) for x in img]
    if len(img[0]) > 3:
        img = [x[:3] for x in img]
    return img

def label_tool(filename1, filename2, eng=None):
    img1 = Image.open(filename1)
    img_mat1 = matlab.uint8(pil_img_to_list(img1))
    img_mat1.reshape((img1.size[0], img1.size[1], 3))

    img2 = Image.open(filename2)
    img_mat2 = matlab.uint8(pil_img_to_list(img2))
    img_mat2.reshape((img2.size[0], img2.size[1], 3))

    if eng is None:
        eng = matlab.engine.start_matlab()
    ret1, ret2 = eng.label_tool(img_mat1, img_mat2, nargout=2)

    return ret1, ret2

if __name__ == '__main__':
    pts1, pts2 = label_tool('data/alita1.png', 'data/alita2.png')
    np.save('data/alita_frame1_pts1.npy', np.array(pts1))
    np.save('data/alita_frame1_pts2.npy', np.array(pts2))
