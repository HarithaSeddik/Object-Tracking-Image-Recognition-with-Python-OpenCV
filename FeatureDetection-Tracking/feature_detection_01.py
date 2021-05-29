# Code that gets input of an image then attempts to locate this image from a videostream using feature detection
# by Khan,Haritha

import cv2
import numpy as np
import time
import math
import imutils

t = time.time()

video = cv2.VideoCapture(0)

while True:

    # Obtain Colored Images

    img1c = cv2.imread("shabnams_bag.jpg")  # ,cv2.IMREAD_GRAYSCALE)
    # print(type(img1c))
    # img2c = cv2.imread("shabnams_bag_test.jpg")#,cv2.IMREAD_GRAYSCALE)
    success, img2c = video.read()

    img2c = imutils.resize(img2c, width=700)

    # Convert to GrayScale
    img1 = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2c, cv2.COLOR_BGR2GRAY)

    # Creat ORB object (Feature Detection Method)
    tracker = cv2.ORB_create(nfeatures=100)

    # Get keypoints and descriptors
    kp1, des1 = tracker.detectAndCompute(img1, None)
    kp2, des2 = tracker.detectAndCompute(img2, None)

    # Match keypoints by descriptors using BRUTE FORCE method
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)  # matching
    # sort shortest distances
    matches = sorted(matches, key=lambda x: x.distance)
    number_of_matches = 10  # choose 10 closest points

    # Get X,Y Coordinates of both the base & test picture
    base_keypoints = np.float32(
        [kp1[m.queryIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)
    test_keypoints = np.float32(
        [kp2[m.trainIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)

    # Create Array for Distances
    array_dis = np.empty(number_of_matches)

    # Get the average x & y coordinates
    avg = np.mean(test_keypoints, axis=0)

    x_avg = avg[0][0]  # average x coordinates
    y_avg = avg[0][1]  # average y coordinates

    count = 0
    # loop over the test picture keypoint coordinates
    for i in test_keypoints:
        x = i[0][0]
        y = i[0][1]
        a = (x - x_avg)**2  # x_dist
        b = (y - y_avg)**2  # y_dist
        array_dis[count] = math.sqrt(a + b)
        count = count+1

    dis_avg = np.mean(array_dis)  # get average dist
    diameter = 1.5*dis_avg  # create virtual diameter for marking

    # Print Circle on test image for marking
    test_img = cv2.circle(img2c, (int(avg[0][0]), int(
        avg[0][1])), int(diameter), (0, 255, 0), 2)

    # print(time.time()-t)
    # print("==========================================================")

    cv2.imshow("image1", img1c)
    cv2.imshow("image2", img2)
    cv2.imshow("test_image", test_img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    # print("===============")

# video.release()
cv2.destroyAllWindows
