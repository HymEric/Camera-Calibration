# -*- coding: utf-8 -*-
# @Time    : 2019/10/12 0012 21:02
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : perspective_transform.py
# @Software: PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('meilan_note6_resized/note01.jpg')
# the four corners of the original image
pts1 = np.float32([[0, 0], [0, 400], [300, 0], [300, 400]])
# the target image four corners: left-up, right-up, left-botom, right-bottom
pts2 = np.float32([[150, 200], [150,600], [450, 200], [450, 600]])
# generate the perspective transformation matrix
M = cv2.getPerspectiveTransform(pts1, pts2)
print(M)
# perspective transformation, the third param is target image size
dst = cv2.warpPerspective(img, M, (600, 800))
plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
plt.show()