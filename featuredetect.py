import cv2
import numpy as np

# Uses SIFT to create descriptors of the keypoints
def SIFT(img, kp):
    print("Finding Features")
    sift = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = sift.compute(img, kp)

    return des1

def SIFTy(img,kp):
    print(("Finding Features"))
    sift = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = sift.detectAndCompute(img,kp)
    return des1, kp1


# Compares the descriptors of image one to image 2 and finds the best 2 matches for each
# ratio test is then used to see if it is an accurate match
def match(des1, des2, kp1, kp2, img1, img2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # finding the two best matches for des1
    goodmatch = []

    #  ratio test
    for m, n in matches:
        if m.distance/n.distance < 0.6:  # 0.7 is the threshold for the ratio test
            goodmatch.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodmatch, img1, flags=2)
    return img3, goodmatch


# This is for my own descriptor and features for part 2

# Object to store keypoint, descriptor and angle of a feature
class Feature:
    def __init__(self, point, descriptor, angle):
        self.keypoint = cv2.KeyPoint(point[1], point[0], 1)
        self.descriptor = descriptor
        self.angle = angle


# TO normalize the descriptor; take the root squared sum and divide each descriptor[i]
def normalize(descriptor):
    sum = np.sqrt(np.sum(np.square(descriptor)))
    for x in range(len(descriptor)):
        descriptor[x] = descriptor[x] / sum
    return descriptor


def createfeatures(keypoints, height, width, img):
    print("Creating Features...")
    features = []
    for x, y in keypoints:
        # To evaluate if there is enough space to make a 16*16 around keypoint
        if x - 8 >= 0 and y - 8 >= 0 and x + 8 < height and y + 8 < width:
            feature = img[x - 8:x + 8, y - 8:y + 8]
            dx, dy = np.gradient(feature)
            dx = cv2.GaussianBlur(dx, (5, 5), 0, 0)
            dy = cv2.GaussianBlur(dy, (5, 5), 0, 0)
            angle = findpeak(hist(dx, dy, 36, 0), 36)  # finds dominant orientation, (can create multiple features)
            descript = featuredescript(dx, dy, angle)  # finds descriptor for feature
            feature = Feature([x, y], descript, angle)  # creates feature object( keypoint + descriptor + angle)
            features.append(feature)
    print("Total of features detected in image: ", len(features))
    return features


# Calculates a descriptor for the feature using SIFT algorithm
def featuredescript(dx, dy, angle):
    descriptor = []
    gridsx = []
    gridsy = []

    # divide 16*16 into 16 4*4 grids
    for x in range(4):
        for y in range(4):
            gridsx.append(dx[x * 4:(x * 4 + 4), y * 4:(y * 4 + 4)])
            gridsy.append(dy[x * 4:(x * 4 + 4), y * 4:(y * 4 + 4)])

    # Calculate an 8 bin histogram for each grid and extend into 1 128d array
    for i in range(len(gridsx)):
        descriptor.extend((hist(gridsx[i], gridsy[i], 8, angle)))  # creating an 128 list/array

    # threshold normalize descriptor
    # divide each in descriptor by the squared root sum of the descriptor
    descriptor = normalize(descriptor)

    # This is for illumination invarience
    for i in range(len(descriptor)):
        if descriptor[i] > 0.2:
            descriptor[i] = 0.2

    # Need to normalize again after performing illumination invarience
    descriptor = normalize(descriptor)

    return descriptor


# finds the index with the max value of the histogram (did I want to try and find multiple with angle)
def findpeak(histogram, size):
    maximum = np.argmax(histogram)  # finds max index
    return maximum * (360 / size)  # need to multiply to find angle size


# Used to find dominant orientation/ key descriptor
# Creates histogram given the images dx,dy, size= number of bins, dom = dominent orientation/ if previously calculated
def hist(dx, dy, size, dom):
    bins = (360/size)
    histogram = np.zeros(size, dtype=np.float32)  # create histogram
    angles = np.arctan2(dy, dx)
    magnitude = np.sqrt(np.square(dy) + np.square(dx))
    angles = np.degrees(angles)

    # Deals with negative angles and calculates new angle given dominent orientation
    for angle in angles:
        for x in range(len(angle)):
            angle[x] = angle[x] - dom   # for rotation invarience
            if angle[x] < 0:
                angle[x] = 360 + angle[x]

    # Place the magnitude at the proper index
    for i in range(len(angles)):
        for j in range(len(angles)):
            index = int((np.floor((angles[i, j]) / bins)) % size)
            histogram[index] += magnitude[i, j]  # number of degrees in each histogram 360/size

    return histogram

#  Function to match the descriptors
def matchfeatures(descript1, descript2, threshold):
    print("Matching Features...")
    low = np.inf
    low2 = np.inf
    keys1 = []
    keys2 = []
    indexmatch = 0

    for each in range(len(descript1)):
        for d in range(len(descript2)):
            dif = difference(descript1[each].descriptor, descript2[d].descriptor)
            if dif < low or dif < low2:
                if dif < low:
                    low = dif
                    indexmatch = d
                else:
                    low2 = dif
        test = ratio(low, low2)
        if test < threshold:  # needs to be less than 0.80 or else error matches
            keys1.append(descript1[each].keypoint)
            keys2.append(descript2[indexmatch].keypoint)
        low = np.inf
        low2 = np.inf
    print("Total Number of Matches: ", len(keys1))
    return keys1, keys2

# Calculating distance between two descriptors
def difference(feat1, feat2):
    ssd = 0
    for i in range(len(feat1)):
        ssd += np.square(feat1[i] - feat2[i])
    return ssd


# ratio test best match divided by second best match
def ratio(ssd1, ssd2):
    return ssd1 / ssd2