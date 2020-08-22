import numpy as np
import cv2
import matplotlib.pyplot as plt


def of_tracing(video_filename, pts, frame_range=(0,10000), displaying=False, out_folder=None):
    result = list()
    result.append(pts)
    cap = cv2.VideoCapture(video_filename)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), minEigThreshold=-1e8)

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # skip the previous frames until the first frame
    for i in range(0, frame_range[0]):
        _, frame = cap.read()

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    idx = frame_range[0] + 1
    while(True):
        if idx >= frame_range[1]:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pts, None, **lk_params)
        print(len(p1))
        result.append(np.array(p1))

        # Select good points
        good_new = p1[st==1]
        good_old = pts[st==1]

        if displaying:
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b),(c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            cv2.imwrite(out_folder+'frame_%d.png'%(idx), img)
            k = cv2.waitKey(300) & 0xff
            if k == 27:
                break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        pts = good_new.reshape(-1, 1, 2)

        idx += 1
    if displaying:
        cv2.destroyAllWindows()
    cap.release()

    return np.array(result)


if __name__ == '__main__':
    pts = np.array([[[173., 514.]], [[219, 1485]], [[263, 777]], [[295, 1706]], [[341, 661]], [[375, 1616]],
    [[459, 496]], [[507, 1454]], [[501, 669]], [[548, 1621]], [[650, 532]], [[713, 1474]]], dtype=np.float32)
    pts = pts[:,:,::-1]

    ret = of_tracing('data/Alita.mp4', pts, frame_range=(0, 10), displaying=True)
    print(len(ret))
