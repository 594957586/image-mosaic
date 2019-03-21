import numpy as np


def calculate_corners(homo, shape):
    height = shape[0]    # height
    width = shape[1]    # width
    pA = np.array([0, 0, 1]).reshape(3, 1)
    pB = np.array([[width, 0, 1]]).transpose()
    pC = np.array([[width, height, 1]]).transpose()
    pD = np.array([[0, height, 1]]).transpose()
    pA = homo.dot(pA)
    pB = homo.dot(pB)
    pC = homo.dot(pC)
    pD = homo.dot(pD)

    return pA/pA[2], pB/pB[2], pC/pC[2], pD/pD[2]


def get_corners(homos, pics):
    transform = np.identity(3, np.float)

    # four corners of each image in the coordinate of the first image
    pAs = []
    pBs = []
    pCs = []
    pDs = []

    bound = [0, pics[0].shape[1], 0, pics[0].shape[0]]  # x_min, x_max, y_min, y_max

    for i in range(1, len(pics)):
        transform = transform.dot(homos[i - 1])  # transform to the first image
        pA, pB, pC, pD = calculate_corners(transform, pics[i].shape)
        bound[0] = min(bound[0], pA[0][0], pD[0][0])
        bound[1] = max(bound[1], pB[0][0], pC[0][0])
        bound[2] = min(bound[2], pA[1][0], pB[1][0])
        bound[3] = max(bound[3], pD[1][0], pD[1][0])

        pAs.append(pA)
        pBs.append(pB)
        pCs.append(pC)
        pDs.append(pD)

    for i in range(len(bound)):
        bound[i] = int(bound[i])
    return bound, pAs, pBs, pCs, pDs
