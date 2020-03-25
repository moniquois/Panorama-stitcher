import numpy as np
import HarrisCorner
import cv2
import featuredetect
import ransac
import stitched


if __name__ == '__main__':
    print("Start of project")
    #Step 1: compute Harris corner detection on Boxes
    img = cv2.imread('project_images/Boxes.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    keypoints = HarrisCorner.computeHarris(gray, 5, 5000)
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]
    img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite('results_images/1a.png', img)

    #Compute Harris Corner Detection on Rainier1 and Rainier2
    img1 = cv2.imread('project_images/Rainier1.png')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img2 = cv2.imread('project_images/Rainier2.png')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    keypoints = HarrisCorner.computeHarris(gray1, 5, 5000)
    keypoints2 = HarrisCorner.computeHarris(gray2, 5, 5000)
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]
    keypoints2 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints2]
    rainier1 = cv2.drawKeypoints(img1, keypoints, None, color=(0, 255, 0), flags=0)
    rainier2 = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)
    cv2.imwrite('results_images/1b.png', rainier1)
    cv2.imwrite('results_images/1c.png', rainier2)

    #Step 2: Matching the interest points between two images
    # Create features/describe using SIFT with keypoints found in Harris Corner Detector
    desc1 = featuredetect.SIFT(img1, keypoints)
    desc2 = featuredetect.SIFT(img2, keypoints2)

    # Compare the features in both images to find matches
    print("Matching features")
    matchedimg, matches = featuredetect.match(desc1, desc2, keypoints, keypoints2, img1, img2)
    cv2.imwrite('results_images/2.png', matchedimg)

    #Step 3: Compute the homography between the images using RANSAC
    good, hom, homInv = ransac.RANSAC(matches, 1500, 5, keypoints, keypoints2)
    matched2 = cv2.drawMatches(img1, keypoints, img2, keypoints2, good, img1, flags=2)
    cv2.imwrite('results_images/3.png', matched2)

    #Step 4: Stitch image together
    stitched.stitch2(img1, img2, hom, homInv, stichedImage=[])



# # this function projects point x,y using homography H
# # returns new projected point
# def project(x1, y1, h):
#
#     x2 = h[0, 0]*x1 + h[0, 1]*y1 + h[0, 2]
#     y2 = h[1, 0]*x1 + h[1, 1]*y1 + h[1, 2]
#     w = h[2, 0]*x1 + h[2, 1]*y1 + h[2, 2]
#
#     x2 = int(x2 / w)
#     y2 = int(y2 / w)
#
#     return x2, y2
#
#
# # Helper function for RANSAC that computes the number of inlier points given a homography H
# # Projects the first point from the match, checks the distance from the real point value
# # if it is below the inlierthreshold it is counted as an inlier
# # returns number of inliers
# def computeInlierCount(h, matches, inlierthreshold):
#     inliers = 0
#     for m in matches:
#         for mat in m:
#             # Get the matching keypoints for each of the images
#             idx1 = mat.queryIdx # index of keypoint from first image
#             idx2 = mat.trainIdx # index of keypoint from second image
#
#             (x1, y1) = keypoints[idx1].pt
#             (x2, y2) = keypoints2[idx2].pt
#             print(x1, y1)
#             print(x2, y2)
#
#             px, py = project(x1, y1, h)
#
#             # check distance from projected point compared to real point
#             if np.sqrt(np.square((px - x2)) + np.square((py-y2))) < inlierthreshold:
#                 inliers += 1
#
#     return inliers
#
# # Finds the keypoints of the inliers and the correct matches
# def findInliers(h, matches, inlierthreshold):
#     inliers1 = []
#     inliers2 = []
#     good = []
#     for m in matches:
#         for mat in m:
#             # Get the matching keypoints for each of the images
#             idx1 = mat.queryIdx
#             idx2 = mat.trainIdx
#
#             (x1, y1) = keypoints[idx1].pt
#             (x2, y2) = keypoints2[idx2].pt
#
#             px, py = project(x1, y1, h)
#
#             if np.sqrt(np.square((px - x2)) + np.square((py-y2))) < inlierthreshold:
#                inliers1.append((x1, y1))
#                inliers2.append((x2, y2))
#                good.append(mat)
#     return inliers1, inliers2, good
#

# # for number of iteration fucntion Chooses 4 randomly selected matches and creates a homography
# # measures the amount of inliers with each homograhpy
# # returns the hom, homInv with the most inliers and the list of matches that are inliers
# def RANSAC(matches, numIterations, inlierThreshold):
#     hom1points = []
#     hom2points = []
#     maxinliers = -np.inf
#     besthom = []
#     kp1, kp2 = keys(matches)
#     for x in range(numIterations):
#         for y in range(4):
#             ranidx = random.randrange(0, len(kp1), 1)
#             hom1points.append(kp1[ranidx])
#             hom2points.append(kp2[ranidx])
#         homography, m = cv2.findHomography(np.asarray(hom1points), np.asarray(hom2points))
#
#         inlier = computeInlierCount(homography, matches, inlierThreshold)
#
#         if(inlier > maxinliers ):
#             maxinliers = inlier
#             besthom = homography
#
#     inlier1, inlier2, goodmatches = findInliers(besthom, matches, inlierThreshold)
#     newhomography, m = cv2.findHomography(np.asarray(inlier1), np.asarray(inlier2))
#
#
#     homInv, m = cv2.findHomography(np.asarray(inlier2), np.asarray(inlier1))
#
#     # get all the inliers
#     return goodmatches, newhomography, homInv

# def keys(matches):
#     # Initialize lists
#     kp1 = []
#     kp2 = []
#
#     for m in matches:
#         # For each match...
#         for mat in m:
#             # Get the matching keypoints for each of the images
#             idx1 = mat.queryIdx
#             idx2 = mat.trainIdx
#
#             # x - columns
#             # y - rows
#             # Get the coordinates
#             (x1, y1) = keypoints[idx1].pt
#             (x2, y2) = keypoints2[idx2].pt
#
#             # Append to each list
#             kp1.append((x1, y1))  # not sure if x1, y1 or y1, x1
#             kp2.append((x2, y2))
#
#     print(kp1)  # keypoints matching for image1
#     print(kp2)  # keypoints matching for image2
#     return kp1, kp2
