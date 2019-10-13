# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 0009 19:42
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : harris_detector.py
# @Software: PyCharm

import cv2
import numpy as np

def test_opencv_harris(img):
    """
    test using opencv api
    :param img: rgb image
    :return:
    """
    img=cv2.resize(img,dsize=(600,400))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=np.float32(gray)

    dst=cv2.cornerHarris(gray,3,3,0.04)
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def harris_detect(img,ksize=3):
    """
    implement by erichym
    :param img: gray image
    :param ksize: Sobel window size
    :return: corners which has the same size with img and the corner pixel value is 255
    """
    k = 0.04  # responding value
    threshold = 0.01  # threshold
    WITH_NMS = False  # whether non-maximum suppression

    # 1. use Sobel to calculate the pixel value gradients of x,y orientation
    h, w = img.shape[:2]

    # Use cv2.CV_16S in case of overflow under uint8
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # 2. Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # 3. filter with Gaussian func for Ix^2,Iy^2,Ix*Iy
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=2)
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4. D=det(M), T=trace(M)
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    # R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 5. threshold and non-maximum suppression for eliminating some fake corners
    R_max = np.max(R)
    # print(R_max)
    # print(np.min(R))
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                # threshold and non-maximum suppression
                if R[i, j] > R_max * threshold and R[i, j] == np.max(
                        R[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)]):
                    corner[i, j] = 255
            else:
                # threshold
                if R[i, j] > R_max * threshold:
                    corner[i, j] = 255
    return corner

if __name__=='__main__':
    img = cv2.imread('meilan_note6_resized/note01.jpg')
    # test_opencv_harris(img)
    img = cv2.resize(img,dsize=(600,400))
    # convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = harris_detect(gray)
    print(dst.shape)  #(400, 600)
    img[dst>0.01*dst.max()] = [0,0,255]
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
