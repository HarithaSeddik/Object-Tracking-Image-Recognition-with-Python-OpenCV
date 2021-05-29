#Code that gets input of an image and matches the object found in another image
#by Khan, Haritha

import cv2
import numpy as np
import time

t= time.time()

img1 = cv2.imread("drone_test_pic.jpg",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("test_picture2.jpg",cv2.IMREAD_GRAYSCALE)

#ORB creation
tracker =cv2.ORB_create(nfeatures=100)

#Get keypoints and descriptors
kp1 , des1 = tracker.detectAndCompute(img1, None)
kp2 , des2 = tracker.detectAndCompute(img2, None)

#Match keypoints by descriptors

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #bruteforce matching method
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

number_of_matches = 5

matching_result= cv2.drawMatches(img1, kp1, img2, kp2, matches[:number_of_matches], None, flags=2)


# Debug print - Draw first 10 matches.

#matches_img = cv2.drawMatches(base_gray, kp_base, test_gray, kp_test, matches[:number_of_matches], flags=2, outImg=base_gray)

#output.debug_show("Matches", matches_img, debug_mode=debug_mode,fxy=fxy,waitkey=True)


# calculate transformation matrix
base_keypoints = np.float32([kp1[m.queryIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)
test_keypoints = np.float32([kp2[m.trainIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)

# print(base_keypoints)
# print(test_keypoints)
# Calculate Homography
h, status = cv2.findHomography(base_keypoints, test_keypoints)



#print(time.time()-t)

cv2.imshow("image1", img1)
cv2.imshow("image2", img2)
cv2.imshow("Matched_Result",matching_result)

# print(time.time()-t)
# print("======================================================")

cv2.waitKey(0)
cv2.destroyAllWindows
