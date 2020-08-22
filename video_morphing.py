import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import cv2
import matlab.engine
from scipy.spatial import Delaunay
import time
from label_tool import label_tool
from optical_flow_tracing import of_tracing

eng = matlab.engine.start_matlab()


def count_frames(filename):
    cap = cv2.VideoCapture(filename)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
    cap.release()
    return i


def get_video_size(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    h, w, c = frame.shape
    cap.release()
    return (h, w)


def select_frame(filename, out_folder, out_prefix, step=30):
    """
    Args:
        filename - the video filename
        out_folder - output selected frames to the out_folder
        out_prefix - output filename prefix
    Return:
        the filename of selected frames
    """
    key_frames_filename = list()
    cap = cv2.VideoCapture(filename)
    i = 0
    while True:
        _, frame = cap.read()
        if not _:
            break
        
        if i%step == 0:
            out_filename = out_folder+out_prefix+'%d.png'%i
            plt.imsave(out_filename, frame[:,:,::-1])
            key_frames_filename.append(out_filename)
        i += 1 
    cap.release()

    return key_frames_filename


def delaunay(points):
    tri = Delaunay(points)
    return tri


def create_tri_mask(sz, pts):
    mask = np.zeros(sz)
    mask = cv2.fillConvexPoly(mask, pts, 1.0, 16, 0)
    return mask


def l2_distance(pts1, pts2):
    """
    Args:
        pts1, pts2: float, (n_frames, n_points, 2)
    Return:
        float: total l2 distance of these frames
    """
    distance = 0
    for i in range(pts1.shape[0]):
        for j in range(pts1.shape[1]):
            distance += ((pts1[i][j][0] - pts2[i][j][0])*(pts1[i][j][0] - pts2[i][j][0]) + \
            (pts1[i][j][1] - pts2[i][j][1])*(pts1[i][j][1] - pts2[i][j][1]))
    return distance


def find_best_interval(cor_pts, interval):
    """
    Args:
        cor_pts: np.float, (n_frames, 2, n_points, 2)
        interval: number of frames for morphing
    Return:
        start_frame index
    """
    best_start = 0
    best_distance = 1e8

    for s in range(0, cor_pts.shape[0]-interval):
        distance = l2_distance(cor_pts[s:s+interval, 0, :, :], cor_pts[s:s+interval, 1, :, :])
        if distance < best_distance:
            best_distance = distance
            best_start = s
    
    return best_start


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

    return np.uint8(final_res)


def video_morphing(project_name, filename1, filename2, configs={}):
    # 0. create a project folder
    root = 'data/' + project_name + '/'
    if not os.path.isdir('data/' + project_name):
        os.mkdir('data/' + project_name)

    # 1. select key frames to fine corresponding points
    if 'frame_step' in configs:
        frame_step = configs['frame_step']
    else:
        frame_step = 10
    total_frames = count_frames(filename1)
    h, w = get_video_size(filename1)
    key_frame_filenames1 = select_frame(filename1, root, '%s1_keyframe_'%project_name, frame_step)
    key_frame_filenames2 = select_frame(filename2, root, '%s2_keyframe_'%project_name, frame_step)

    # 2. manually label corresponding points for selected key frames
    cor_pts = list()
    for i in range(len(key_frame_filenames1)):
        pts1, pts2 = label_tool(key_frame_filenames1[i], key_frame_filenames2[i], eng)
        pts1, pts2 = np.array(pts1), np.array(pts2)
        pts1, pts2 = np.transpose(pts1), np.transpose(pts2)
        cor_pts.append(np.array([pts1, pts2]))
    cor_pts = np.array(cor_pts)
    np.save(root+'cor_pts.npy', cor_pts)
    
    # dubging in testing
    #cor_pts = np.load(root+'cor_pts.npy')

    # 3. use optical flow to trace the corresponding points for the rest of frames
    cor_pts_all = list()
    j = 0
    for i in range(0, total_frames, frame_step):
        res1 = of_tracing(filename1, np.expand_dims(cor_pts[j][0], axis=1).astype(np.float32), (i, i+frame_step), displaying=False, out_folder=root)
        res2 = of_tracing(filename2, np.expand_dims(cor_pts[j][1], axis=1).astype(np.float32), (i, i+frame_step), displaying=False, out_folder=root)
        res = np.concatenate([res1, res2], axis=2)
        cor_pts_all.append(res)
        j += 1
    cor_pts_all = np.concatenate(cor_pts_all, axis=0)

    # 4. find best interval via spatial-temporal alignment
    interval = configs['interval']
    start_idx = find_best_interval(cor_pts_all, interval)
    #start_idx = np.random.randint(0, total_frames-interval-1)
    print(start_idx)

    # 5. morphing each frame
    out = cv2.VideoWriter(root+'%s_result.mp4'%project_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    cap1 = cv2.VideoCapture(filename1)
    cap2 = cv2.VideoCapture(filename2)

    # 5.a. copy each frame till start index
    cnt = 0
    for i in range(start_idx):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        out.write(frame1)
        cnt += 1

    # 5.b. morphing each frame in the interval
    fac = 1.0/(interval-1)

    # create delaunay triangle at begining
    p1 = cor_pts_all[0,:,0,:]
    p1 = np.concatenate((p1, np.array([[0, 0], [0, h-1], [w-1,0], [w-1, h-1]])), axis=0)
    tri = delaunay(p1)
    for alpha in np.arange(0, 1, fac):
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()

        if cnt == start_idx:
            mid_frame = frame1
        else:
            p1 = cor_pts_all[cnt,:,0,:]
            p2 = cor_pts_all[cnt,:,1,:]

            # add 4 conner to the corresponding points
            p1 = np.concatenate((p1, np.array([[0, 0], [0, h-1], [w-1,0], [w-1, h-1]])), axis=0)
            p2 = np.concatenate((p2, np.array([[0, 0], [0, h-1], [w-1,0], [w-1, h-1]])), axis=0)

            # create delaunay triangle for each frame
            #tri = delaunay(p1)

            # morphing frame
            mid_frame = morphing_frame(frame1, frame2, p1, p2, tri.simplices, 1-alpha)

        #cv2.imwrite('data/result_alita3/frame_%d.png'%cnt, mid_frame)
        out.write(mid_frame)
        cnt += 1
    _, frame2 = cap2.read()
    #cv2.imwrite('data/result_alita3/frame_%d.png'%cnt, frame2)
    out.write(frame2)

    # 5.c. copy all the rest frames to the output
    while True:
        ret, frame2 = cap2.read()
        if not ret:
            break
        out.write(frame2)

    cap1.release()
    cap2.release()
    out.release()


if __name__ == '__main__':
    filename1 = 'dance/sugar1ss.mp4'
    filename2 = 'dance/sugar2s.mp4'
    project_name = 'sugar'

    configs = {
        'frame_step': 60,
        'interval': 40,
        'output_frame_rate': 20,
    }
    result = video_morphing(project_name, filename1, filename2, configs)
