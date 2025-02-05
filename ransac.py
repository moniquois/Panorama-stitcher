import cv2
import numpy as np
import random


# function projects point x,y using homography H
# returns new projected point
def project(x1, y1, h):

    x2 = h[0, 0]*x1 + h[0, 1]*y1 + h[0, 2]
    y2 = h[1, 0]*x1 + h[1, 1]*y1 + h[1, 2]
    w = h[2, 0]*x1 + h[2, 1]*y1 + h[2, 2]

    x2 = int(x2 / w)
    y2 = int(y2 / w)

    return x2, y2

# for number of iterations, function chooses 4 randomly selected matches
# creates a homography with them
# measures the amount of inliers with each homograhpy
# returns the hom, homInv with the most inliers and the list of matches that are inliers
def RANSAC(matches, numIterations, inlierThreshold, keypoints, keypoints2):
    hom1points = []
    hom2points = []
    maxinliers = -np.inf
    hom = []
    kp1, kp2 = keys(matches, keypoints, keypoints2)  # finds the keypoints from the matches
    for x in range(numIterations):
        for y in range(4):
            ranidx = random.randrange(0, len(kp1), 1)
            hom1points.append(kp1[ranidx])
            hom2points.append(kp2[ranidx])

        homography, m = cv2.findHomography(np.asarray(hom1points), np.asarray(hom2points))  # finds homography
        inlier = computeInlierCount(homography, matches, inlierThreshold, keypoints, keypoints2)  # finds num of inliers

        if inlier > maxinliers:
            maxinliers = inlier
            hom = homography

    # gets inliers and good matches with homography
    inlier1, inlier2, goodmatches = findInliers(hom, matches, inlierThreshold, keypoints, keypoints2)

    hom, m = cv2.findHomography(np.asarray(inlier1), np.asarray(inlier2))
    homInv, m = cv2.findHomography(np.asarray(inlier2), np.asarray(inlier1))

    # get all the inliers
    return goodmatches, hom, homInv


# Helper function for RANSAC that computes the number of inlier points given a homography H
# Projects the first point from the match, checks the distance from the real point value
# if it is below the inlierthreshold it is counted as an inlier
# returns number of inliers
def computeInlierCount(h, matches, inlierthreshold, keypoints, keypoints2):
    inliers = 0
    for m in matches:
        for mat in m:
            # Get the matching keypoints for each of the images
            idx1 = mat.queryIdx  # index of keypoint from first image
            idx2 = mat.trainIdx  # index of keypoint from second image

            (x1, y1) = keypoints[idx1].pt
            (x2, y2) = keypoints2[idx2].pt

            px, py = project(x1, y1, h)

            # check distance from projected point compared to real point
            if np.sqrt(np.square((px - x2)) + np.square((py-y2))) < inlierthreshold:
                inliers += 1

    return inliers

# Finds the keypoints of the inliers and the correct matches
def findInliers(h, matches, inlierthreshold, keypoints, keypoints2):
    inliers1 = []
    inliers2 = []
    good = []

    for m in matches:
        for mat in m:
            # Get the matching keypoints for each of the images
            idx1 = mat.queryIdx
            idx2 = mat.trainIdx

            (x1, y1) = keypoints[idx1].pt
            (x2, y2) = keypoints2[idx2].pt

            px, py = project(x1, y1, h)

            if np.sqrt(np.square((px - x2)) + np.square((py-y2))) < inlierthreshold:
                inliers1.append((x1, y1))
                inliers2.append((x2, y2))
                good.append(mat)

    return inliers1, inliers2, good


# Finds the keypoints in both images related to the chosen matches
def keys(matches, keypoints, keypoints2):
    # Initialize lists
    kp1 = []
    kp2 = []

    for m in matches:
        for match in m:
            # Get the matching keypoints for each of the images
            idx1 = match.queryIdx  # image1
            idx2 = match.trainIdx  # image2

            (x1, y1) = keypoints[idx1].pt
            (x2, y2) = keypoints2[idx2].pt

            # Append to each list
            kp1.append((x1, y1))
            kp2.append((x2, y2))

    return kp1, kp2

