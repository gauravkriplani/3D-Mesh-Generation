import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Camera number to easily switch cameras
camera_number = 1 

calibimgfiles = f'calib/frame_C{camera_number}_*.png'
resultfile = f'calibration_C{camera_number}.pickle'

# checkerboard coordinates in 3D
checkerboard_size = (8, 6)
square_size = 2.8

# checkerboard coordinates in 3D
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = 2.8*np.mgrid[0:8, 0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob(calibimgfiles)

if len(images) == 0:
    print("No images found.")
    exit()

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Display image with the corners overlayed
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# now perform the calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

#  Store the extrinsic parameters (rotation and translation vectors)
extrinsics = []
for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvec)
    extrinsics.append({
        "R": R,
        "t": tvec
    })

# save the results out to a file for later use
calib = {}
calib["fx"] = K[0][0]
calib["fy"] = K[1][1]
calib["cx"] = K[0][2]
calib["cy"] = K[1][2]
calib["dist"] = dist
calib["extrinsics"] = extrinsics
fid = open(resultfile, "wb" ) 
pickle.dump(calib,fid)
fid.close()