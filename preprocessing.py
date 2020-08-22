import numpy as np
import cv2

def crop_video(box=None):
    cap = cv2.VideoCapture('dance/sugar1s.mp4')
    out = cv2.VideoWriter('dance/sugar1ss.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (1280, 720))

    cnt = 0
    while True:
        _, frame = cap.read()
        if not _:
            break
        h, w = frame.shape[:2]

        #frame = cv2.resize(frame,(int(1280),int(720)))

        if cnt >= 0 and cnt < 120:
            out.write(frame)
            cv2.imshow("Frame", frame)
            if cnt == 100:
                cv2.imwrite('dance/frame1.png', frame)

            key = cv2.waitKey(33)
            if key == 33:
                break

        cnt += 1

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    crop_video()
