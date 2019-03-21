import cv2


def get_pictures(num):
    pics = []
    for i in range(num):
        pic = cv2.imread("{}.jpg".format(i+1))
        pic = cv2.resize(pic, (1280, 960))
        if pic is None:
            return None
        pics.append(pic)
    return pics

