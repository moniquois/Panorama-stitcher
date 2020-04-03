import numpy as np
import HarrisCorner
import cv2
import featuredetect
import ransac
import stitched

#  function to stitch rest of rainier photos together
def panorama(img, img2):

    desc1, keypoints = featuredetect.SIFTy(img, None)
    desc2, keypoints2 = featuredetect.SIFTy(img2, None)

    matchedimg, matches = featuredetect.match(desc1, desc2, keypoints, keypoints2, img, img2)

    good, hom, homInv = ransac.RANSAC(matches, 1500, 5, keypoints, keypoints2)
    matched = cv2.drawMatches(img, keypoints, img2, keypoints2, good, img, flags=2)
    #cv2.imwrite('results_images/match.png', matched)
    finalresult = stitched.stitch(img, img2, hom, homInv, stitchedImage=[])
    cv2.imwrite('results_images/AllStitched.png', finalresult)
    return finalresult

def completePicture(stitchedimage):
    img1 = cv2.imread('project_images/Rainier3.png')
    img2 = cv2.imread('project_images/Rainier4.png')
    img3 = cv2.imread('project_images/Rainier5.png')
    img4 = cv2.imread('project_images/Rainier6.png')

    stitchedimage = panorama(stitchedimage, img1)
    stitchedimage = panorama(stitchedimage, img2)
    stitchedimage = panorama(stitchedimage, img3)
    stitchedimage = panorama(stitchedimage, img4)

if __name__ == '__main__':
    print("Start of project")

    #Step 1: compute Harris corner detection on Boxes
    img = cv2.imread('project_images/Boxes.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    keypoints, img = HarrisCorner.fdkeypoints(img, gray)
    cv2.imwrite('results_images/1a.png', img)

    # Read in images and dimensions
    img1 = cv2.imread('project_images/Rainier1.png')
    height1, width1, c = img1.shape
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img2 = cv2.imread('project_images/Rainier2.png')
    height2, width2, c  = img2.shape
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # This is my own descriptor for Step 2: not used for parts 3 & 4
    # # Create features using SIFT around Keypoints; returns array of descriptors
    keypoints3 =HarrisCorner.computeHarris(gray1, 5, 5000)
    keypoints4 = HarrisCorner.computeHarris(gray2, 5, 5000)
    descriptors1 = featuredetect.createfeatures(keypoints3, height1, width1,  gray1)
    descriptors2 = featuredetect.createfeatures(keypoints4, height2, width2, gray2)
    keys1, keys2 = featuredetect.matchfeatures(descriptors1, descriptors2, 0.3)  # use 0.5 for higher threshold / above 0.75 creates false positives


    # Step 2: Matching the interest points between two images
    # Create features/describe using SIFT with keypoints found in Harris Corner Detector
    keypoints, keyimg = HarrisCorner.fdkeypoints(img1, gray1)
    keypoints2, keyimg2 = HarrisCorner.fdkeypoints(img2, gray2)
    cv2.imwrite('results_images/1b.png', keyimg)
    cv2.imwrite('results_images/1c.png', keyimg2)
    desc1 = featuredetect.SIFT(img1, keypoints)
    desc2 = featuredetect.SIFT(img2, keypoints2)

    # Compare the features in both images to find matches
    print("Matching features")
    matchedimg, matches = featuredetect.match(desc1, desc2, keypoints, keypoints2, img1, img2)
    cv2.imwrite('results_images/2.png', matchedimg)

    # Step 3: Compute the homography between the images using RANSAC
    good, hom, homInv = ransac.RANSAC(matches, 1500, 5, keypoints, keypoints2)
    matched2 = cv2.drawMatches(img1, keypoints, img2, keypoints2, good, img1, flags=2)
    cv2.imwrite('results_images/3.png', matched2)

    # Step 4: Stitch images together
    stitchedImage = stitched.stitch(img1, img2, hom, homInv, stitchedImage=[])
    cv2.imwrite('results_images/4.png', stitchedImage)

    # Stitching the rest together
    completePicture(stitchedImage)
