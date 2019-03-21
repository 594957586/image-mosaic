import cv2
import cv2.xfeatures2d as contrib
import numpy as np
import pickle
import picture_preprocessing


def get_features(pic, hessian_threshold=100, show_key_point=False):
    surf = contrib.SURF_create(hessian_threshold)
    key, des = surf.detectAndCompute(pic, None)
    if show_key_point:
        kp_image = np.zeros(pic.shape[:-1]).astype(np.uint8)
        kp_image = cv2.drawKeypoints(kp_image, key, None, [255, 255, 255])
        cv2.namedWindow("key point", cv2.WINDOW_NORMAL)
        cv2.imshow("key point", kp_image)
        cv2.waitKey()
    return key, des


def get_good_features(pics):
    keys = []
    dess = []
    for pic in pics:
        key, des = get_features(
            pic=pic,
            hessian_threshold=800,
            show_key_point=False
        )
        keys.append(key)
        dess.append(des)
    return keys, dess

