# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 0008 18:26
# @Author  : Erichym
# @Email   : 951523291@qq.com
# @File    : calibration_opencv.py
# @Software: PyCharm
import numpy as np
import cv2
import glob
import os

# termination criteria
# setting the parameters of searching sub-pixel corners, applying the maximum looping iteration 30 for stopping rule and max error tolerance 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
board_size=30
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('meilan_note6_resized/*.jpg')
img_root='meilan_note6_resized'
img_names=os.listdir(img_root)
img_paths=[os.path.join(img_root,f) for f in img_names]
print(len(img_paths))

for fname in img_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp*board_size)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey()

cv2.destroyAllWindows()

# print(objpoints)
# print(imgpoints)
print("objpoints len:{}:".format(len(objpoints)))
print("imgpoints len:{}".format(len(imgpoints)))

# ret the min value of max likelihood function
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("intrinsic parameters: {}".format(mtx))
print("extrinsic parameters: \nR:{},\nT:{}".format(rvecs,tvecs))
print("distortion parameters: {}".format(dist))


# img = cv2.imread('chessboards_imgs/left12.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#
# # Using cv2.undistort()
# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult_undistort.png',dst)
#
# # Using remapping it will crop image little
# # undistort
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult_remap.png',dst)