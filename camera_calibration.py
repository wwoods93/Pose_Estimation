######################################################################################
###     camera_calibration.py
###     University of Akron NASA Robotic Mining Team
###     Source:     OpenCV calibration tutorial
###     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
###     Date:       11.13.2020
###     Authors:    Wilson Woods
###                 David Klett
######################################################################################

import numpy as np
import cv2 as cv
import glob
from collections import namedtuple

# number of inner corners on chessboard (row, column)
num_corners(4, 3)
coord_pair = namedtuple('coord_pair', ['row', 'col'])
num_corners = coord_pair(row=4, col=3)

# Set termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_points = np.zeros((num_corners.row * num_corners.col, 3), np.float32)
obj_points[:, :2] = np.mgrid[0:num_corners.row, 0:num_corners.col].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
obj_points_set = []  # 3-D points in real world space
img_points_set = []  # 2-D points in image plane
training_set = glob.glob('*.jpg')

for jpg_image in training_set:
    image = cv.imread(jpg_image)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Find chessboard corners
    corners_found, corners = cv.findChessboardCorners(gray_image, (num_corners.row, num_corners.col), None)
    # If found, add object points, image points (after refining them)
    if corners_found:
        # add 3-D points to object points array
        obj_points_set.append(object_points)
        # Refine corner location to sub-pixel level accuracy
        refined_corners = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
        img_points_set.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(image, (num_corners.row, num_corners.col), refined_corners, corners_found)
        resized_image = cv.resize(image, (960, 540))                    # Resize image
        cv.imshow("output", resized_image)                              # Show image
        # cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()
# Calibrate the camera
rms_error, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    obj_points_set, img_points_set, gray_image.shape[::-1], None, None)

print("RMS Re-projection Error:")
print(rms_error)
print("mtx:")
print(mtx)
print("dist:")
print(dist)
print("rvecs")
print(rvecs)
print("tvecs:")
print(tvecs)

np.savez('video.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
