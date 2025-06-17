import cv2
import math
import numpy as np
from runs.tools import util_fn

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255],
                    [255, 0, 0], [255, 255, 255]])

# 23 joints
skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
            [3, 5], [4, 6], [15, 17], [17, 18], [15, 19], 
            [16, 20], [20, 21], [16, 22]]

pose_link_color = palette[[
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]]

def imshow_keypoints(img, keypoints,
                     radius=4, 
                     thickness=1,):
    img_h, img_w, _ = img.shape
    for kid, kpt in enumerate(keypoints):
        if kpt[2] == 0.0:
            continue
        
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
    
        color = tuple(int(c) for c in pose_kpt_color[kid])
        cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)
    
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]))
        pos2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]))

        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                        and pos2[1] > 0 and pos2[1] < img_h
                        and keypoints[sk[0], 2] > 0
                        and keypoints[sk[1], 2] > 0):
            
            color = tuple(int(c) for c in pose_link_color[sk_id])
            cv2.line(img, pos1, pos2, color, thickness=thickness)
    
    return img


def imshow_bbox(img, center=None, scale=None, xyxy=None, thickness=3):
    if xyxy is None:
        xyxy = util_fn.cs2xyxy(center, scale)
    
    pt1 = (int(xyxy[0]), int(xyxy[1]))
    pt2 = (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=thickness)
    return img