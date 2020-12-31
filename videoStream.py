######################################################################################
###     videoStream.py
###     University of Akron NASA Robotic Mining Team
###     Source:     OpenCV pose estimation tutorial
###     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
###     Date:       11.13.2020
###     Authors:    Wilson Woods
###                 David Klett
######################################################################################

import numpy as np
import cv2 as cv
import glob
import math
import time
import imutils
from collections import namedtuple
from queue import Queue

# buffer for moving_avg
BUFFER_SIZE = 50
sum = 0
x_sum = 0
y_sum = 0
buffer = Queue(maxsize = BUFFER_SIZE)
xy_buffer = Queue(maxsize = BUFFER_SIZE)

# Load calibration data
with np.load('video.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def location(corners):
    percentage = corners[0][0][0]/1280
    width = abs(corners[0][0][0] - corners[8][0][0])
    height = abs(corners[0][0][1] - corners[3][0][1])
    return percentage, width, height


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the target to the camera
	return (knownWidth * focalLength) / perWidth


def moving_avg_1(cam_output):
        global sum
        if buffer.full():
            sum -= buffer.get()
        buffer.put(cam_output)
        sum += cam_output
        return sum / BUFFER_SIZE


def moving_avg_2(cam_output):
        global x_sum, y_sum
        if xy_buffer.full():
            out = xy_buffer.get()
            x_sum -= out[0]
            y_sum -= out[1]
        xy_buffer.put(cam_output)
        x_sum += cam_output[0]
        y_sum += cam_output[1]
        return x_sum / BUFFER_SIZE, y_sum / BUFFER_SIZE

coord_pair = namedtuple('coord_pair', ['row', 'col'])
num_corners = coord_pair(row=4, col=3)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280) #800, 1280
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720) #600, 720
cap.set(cv.CAP_PROP_BUFFERSIZE, 0)

# pixel length = 212.6, distance =
known_distance = 67 # cm
known_length = 15.5 # cm
pixel_length = 212.6
focal_length = (pixel_length * known_distance) / known_length


while(True):
    # Capture frame-by-frame
    read_success, frame = cap.read()
    # Our operations on the frame come here
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners_found, raw_corners = cv.findChessboardCorners(gray_image, (num_corners.row, num_corners.col), None)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((num_corners.col * num_corners.row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners.row,0:num_corners.col].T.reshape(-1,2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    if corners_found:
        refined_corners = cv.cornerSubPix(gray_image, raw_corners, (11,11), (-1,-1), criteria)
        length = abs(refined_corners[0][0][1] - refined_corners[3][0][1])
        #focal_lenth = (length[0]*known_distance)/known_width
        distance = distance_to_camera(known_length, focal_length, length)

        # Find the rotation and translation vectors.
        try:
            ret, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, refined_corners, mtx, dist) #try Ransac
        except:
            print("error")
            continue
        #r = R.from_rotvec(3,rvecs)
        # Convert 3x1 rotation vector to rotation matrix for further computation            
        rotation_matrix, jacobian = cv.Rodrigues(rvecs)
        tvecs_new = -np.matrix(rotation_matrix).T * np.matrix(tvecs)

        # Projection Matrix
        pmat = np.hstack((rotation_matrix, tvecs)) # [R|t]
        roll, pitch, yaw = cv.decomposeProjectionMatrix(pmat)[-1]
        
        x_distance = math.cos(roll * math.pi / 180) * distance
        y_distance = abs(math.sin(roll * math.pi / 180) * distance)
        xy_pair = (x_distance, y_distance)
        x_avg, y_avg = moving_avg_2(xy_pair)
        dist_avg = moving_avg_1(distance)
        
        print("X DISTANCE:")
        print(x_avg)
        print("Y DISTANCE:")
        print(y_avg)
        # print("DISTANCE:")
        # print(dist_avg)
        # print('Roll: {:.2f}\tPitch: {:.2f}\tYaw: {:.2f}'.format(float(roll), float(pitch), float(yaw)))

        # # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        try:
            img = draw(frame, refined_corners, imgpts)
        except:
            print("error in draw function")
            continue

    # Show image 
    frame = imutils.resize(frame, width=800)
    cv.imshow("output", frame)  
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # # Display the resulting frame
    #frame = cv.resize(frame, (540,960))  
    #cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # k = cv2.waitKey(0) & 0xFF
    # if k == ord('s'):
    #     cv2.imwrite(fname[:6]+'.png', frame)

cap.release()
cv2.destroyAllWindows()
