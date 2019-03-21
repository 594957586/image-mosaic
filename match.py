import cv2
import numpy as np
import feature
import picture_preprocessing


def good_match(des_1, des_2, left_num=50):
    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors=des_1,
                            trainDescriptors=des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:left_num]


def get_transform_matrix(keys, dess, pics, recalculate=True, show_match=False):
    homos = []
    if recalculate:
        for i in range(1, len(keys)):
            good_matches = good_match(dess[i-1], dess[i], 50)
            if show_match:
                match_pic = cv2.drawMatches(
                    pics[i-1], keys[i-1], pics[i], keys[i], good_matches[:50],
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                )
                cv2.namedWindow("match pic", cv2.WINDOW_NORMAL)
                cv2.imshow("match pic", match_pic)
                cv2.waitKey()

            points_in_pic_1 = []
            points_in_pic_2 = []
            for j in range(len(good_matches)):
                points_in_pic_1.append(keys[i-1][good_matches[j].queryIdx].pt)
                points_in_pic_2.append(keys[i][good_matches[j].trainIdx].pt)

            points_in_pic_1 = np.array(points_in_pic_1)
            points_in_pic_2 = np.array(points_in_pic_2)

            homo, mask = cv2.findHomography(
                srcPoints=points_in_pic_2,
                dstPoints=points_in_pic_1,
                method=cv2.RANSAC
            )
            homos.append(homo)
            np.save("homo_{}.npy".format(i), homo)
    else:
        for i in range(1, len(keys)):
            homo = np.load("homo_{}.npy".format(i))
            homos.append(homo)
    return homos


if __name__ == "__main__":
    pics = picture_preprocessing.get_pictures(5)
    keys, dess = feature.get_good_features(pics)
    get_transform_matrix(keys, dess, pics, recalculate=True, show_match=True)
