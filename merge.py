import numpy as np
import math


def distance(l_p1, l_p2, p):
    l_p1.reshape(3)
    l_p2.reshape(3)
    p.reshape(3)

    S = l_p1[0]*(l_p2[1] - p[1]) + l_p2[0]*(p[1] - l_p1[1]) + p[0]*(l_p1[1] - l_p2[1])
    l = math.sqrt((l_p1[0] - l_p2[0])**2 + (l_p1[1] - l_p2[1])**2)
    return S/l


def merge(src, dst, p1, p2, p3, p4):
    """
    :param src: source picture
    :param dst: added on picture
    :param p1: source right-top point
    :param p2: source right-bottom point
    :param p3: dst left-top point
    :param p4: dst left-bottom point
    :return: merged image
    """
    for x in range(src.shape[1]):
        for y in range(src.shape[0]):
            p = np.array([x, y, 1]).astype(np.float)
            if (src[y][x] != 0).all() and (dst[y][x] != 0).all():
                r = 0.5
                src[y][x] = src[y][x]*r + dst[y][x]*(1 - r)
            elif (src[y][x] == 0).all() and (dst[y][x] != 0).all():
                src[y][x] = dst[y][x]
        # print(float(x)/src.shape[1])
    return src


