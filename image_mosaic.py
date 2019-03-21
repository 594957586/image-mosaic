import cv2
import numpy as np
import picture_preprocessing
import feature
import match
import image_transform
import merge


def save_ps(pAs, pBs, pCs, pDs):
    with open("ps.dat", "w") as file:
        for i in range(len(pAs)):
            pAs[i] = pAs[i][:, 0]
            pBs[i] = pBs[i][:, 0]
            pCs[i] = pCs[i][:, 0]
            pDs[i] = pDs[i][:, 0]
            line = "{} {} {} {} {} {} {} {}\n".format(
                pAs[i][0], pAs[i][1],
                pBs[i][0], pBs[i][1],
                pCs[i][0], pCs[i][1],
                pDs[i][0], pDs[i][1])
            file.write(line)


if __name__ == "__main__":
    num = 4
    pics = picture_preprocessing.get_pictures(num)
    keys, dess = feature.get_good_features(pics)
    homos = match.get_transform_matrix(keys, dess, pics)
    bound, pAs, pBs, pCs, pDs = image_transform.get_corners(homos, pics)
    save_ps(pAs, pBs, pCs, pDs)
    transform = np.identity(3, np.float)
    dst = np.zeros((bound[3], bound[1], 3)).astype(np.uint8)
    dst[0:pics[0].shape[0], 0:pics[0].shape[1], :] = pics[0]    # origin image
    cv2.imwrite("result_0.jpg", dst)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    for i in range(1, num):
        transform = transform.dot(homos[i-1])

        tmp = cv2.warpPerspective(pics[i], transform, (bound[1], bound[3]))

        # dst = merge.merge(dst, tmp, pBs[i-1], pCs[i-1], pAs[i], pDs[i])

        cv2.imshow("result", tmp)
        cv2.imwrite("result_{}.jpg".format(i), tmp)
        cv2.waitKey()


