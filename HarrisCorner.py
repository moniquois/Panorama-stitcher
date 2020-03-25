import numpy as np
import cv2


#   Function to compute the Harris Corner algorithm to find identify keypoints in image
#   Returns a list of keypoints
def computeHarris(img, window, threshold):
    print("Finding Keypoints...")
    height, width = img.shape

    #   take derivatives of image
    dx, dy = np.gradient(img)

    # Get squares of derivatives
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # Smooth each with gaussian
    dx2 = cv2.GaussianBlur(dx2, (5, 5), 0, 0)
    dy2 = cv2.GaussianBlur(dy2, (5, 5), 0, 0)
    dxy = cv2.GaussianBlur(dxy, (5, 5), 0, 0)

    #   take a window over the derivatives, sum and then calculate the corner strength function
    win = int(window / 2)
    rangec = np.zeros((height, width), dtype=int)
    keypoints = []
    for x in range(win, height - win):
        for y in range(win, width - win):
            newdx = dx2[x - win:x + win + 1, y - win:y + win + 1]
            newdy = dy2[x - win:x + win + 1, y - win:y + win + 1]
            newxy = dxy[x - win:x + win + 1, y - win:y + win + 1]
            newdx = newdx.sum()
            newdy = newdy.sum()
            newxy = newxy.sum()

            # Calculate corner strength function(C)
            # Calculate det and trace
            det = newdx * newdy - newxy * newxy
            trace = newdx + newdy
            c = det / (trace + 0.001)  # add very small number to denominator so we never divide by 0
            rangec[x, y] = c

    # Check if C is the maximum in a 3*3 neighbourhood
    for x in range(1, height):
        for y in range(1, width):
            neighbour = rangec[x - 1:x + 2, y - 1:y + 2]  # 3 by 3 neighbour hood max
            if rangec[x, y] == np.max(neighbour) and rangec[x, y] > threshold:
                keypoints.append((x, y))  # storing values of keypoints that are useful

    # return keypoints that are detected as corners
    print("Total of keypoints found: ", len(keypoints))
    return keypoints
