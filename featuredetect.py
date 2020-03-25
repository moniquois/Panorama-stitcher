import cv2


# Finds features using keypoints detected from Harris Corner detector and uses SIFT to describe them
def SIFT(img, kp):
    print("Finding Features")
    #   use this function to find the features using opencv functions
    sift = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = sift.compute(img, kp)

    return des1


# compares the descriptors to find the best matches
def match(des1, des2, kp1, kp2, img1, img2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # finding the two best matches for des1
    good = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img1, flags=2)
    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, img1, flags=2)
    return img3, good
