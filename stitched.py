import cv2
import numpy as np
import ransac

# STEP 4 - Stitch the images together using the computed homography and inverse homography
# function finds stitches together two images by projecting all points from the newimage that
# and checking if they are contained in image2; then they are added/blended
def stitch(image1, image2, hom, homInv, stitchedImage):
    print("Stitching image")
    height1, width1, c = image1.shape
    height2, width2, c = image2.shape

    # calculates the dimension of combining the two images
    stitchedImage, minheight, minwidth = resizeimage(image1, image2, homInv)
    stitchheight, stitchwidth, c = stitchedImage.shape

    # Copy image1 into new image at proper position
    stitchedImage[minheight:height1+minheight, minwidth:width1+minwidth] = image1
    #cv2.imwrite('results_images/4.png', stitchedImage)

    # project each point in new image if point lies within image2 boundaries add or blend pixel value
    for x in range(stitchwidth):
        for y in range(stitchheight):
            newx, newy = ransac.project(x-minwidth, y-minheight, hom)
            if newx<=width2 and newy<=height2 and newx>=0 and newy>=0:
                val = cv2.getRectSubPix(image2, (1, 1), (newx, newy))
                stitchedImage[y, x] = val

    #cv2.imwrite('results_images/4.png', stitchedImage)
    return stitchedImage

# finds the needed size of stitching the two images together
def resizeimage(image1, image2, homInv):
    height1, width1, c = image1.shape
    height2, width2, c = image2.shape

    stitchheight = height1
    stitchwidth = width1

    # x is height, y is width
    # Find the projected 4 corners of image 2 onto image1
    x1, y1 = ransac.project(0, 0, homInv)  # top left corner
    x2, y2 = ransac.project(width2, height2, homInv)  # upper right corner
    x3, y3 = ransac.project(width2, 0, homInv)
    x4, y4 = ransac.project(0, height2, homInv)

    minheight = min(y1, y3) #find the minheight(if corner goes into the negative)
    maxheight = max(y2, y4)
    maxwidth = max(x2, x3)
    minwidth = min(x1, x4)
    print("minheight: ", minheight)
    print("minwidth: ", minwidth)
    print("maxheight: ", maxheight)
    print("mawidth: ", maxwidth)

    if maxheight > height1:
        stitchheight = maxheight

    if minheight < 0:
        stitchheight -= minheight  # add height to the new image
        minheight = -minheight
    elif minheight > 0:
        minheight = 0

    if maxwidth > width1:

        stitchwidth = maxwidth

    if minwidth < 0:
        stitchwidth -= minwidth #addwidth to new image
        minwidth = -minwidth
    elif minwidth > 0:
        minwidth = 0



    stitchedImage = np.zeros(shape=[stitchheight, stitchwidth, 3], dtype=np.uint8)
    print(stitchwidth, stitchheight)
    return stitchedImage, minheight, minwidth

