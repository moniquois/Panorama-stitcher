import cv2
import numpy as np
import ransac

# STEP 4 - Stitch the images together using the computed homography

def stitch(image1, image2, hom, homInv, stichedImage):
    print("Stitching image")
    stichedheight, stitchedwidth, c = image1.shape
    height, width, c = image1.shape
    height2, width2, c = image2.shape
    #
    # x2, y2 = ransac.project(height2, width2, homInv)  #  568 across 522 down
    # print("projected size: lower right corner ", x2, y2)
    # if x2 > height1:
    #     height1 = x2
    # if y2 > width1:
    #     width1 = y2
    # x2, y2 = ransac.project(0, 0, homInv)  # 156 6 -  156 across (width) - 6(height) down
    # print("Projected size: upper left corner ", x2, y2)
    # if x2 > height1:
    #     height1 = x2
    # if y2 > width1:
    #     width1 = y2
    # x2, y2 = ransac.project(height2, 0, homInv)  # upper right
    # print("Projected size: upper right corner ", x2, y2) # -43 up and 554 across
    # if x2 > height1:
    #     height1 = x2
    # if y2 > width1:
    #     width1 = y2
    # x2, y2 = ransac.project(0, width2, homInv)
    # print("Projected size: lower left corner ", x2, y2)  # 174 across 482 down
    # if x2 > height1:
    #     height1 = x2
    # if y2 > width1:
    #     width1 = y2


    stichedImage = np.zeros(shape=[1000, 1000, 3], dtype=np.uint8)
    #blank_image = np.zeros(shape=[width1 + 43, height1,  3], dtype=np.uint8)
    stichedImage[0: height, 0: width] = image1
    cv2.imwrite('results_images/4.png', stichedImage)


    # for each value in the second image project onto blank_image
    for x in range(width2):
        for y in range(height2):
            newx, newy = ransac.project(x, y, homInv)

            stichedImage[newy, newx] = image2[y, x]
    cv2.imwrite('results_images/4.png', stichedImage)

def stitch2(image1, image2, hom, homInv, stichedImage):
    # calculate dimension of stitched image
    # copy image 1 into proper place
    # project stitched image pixel onto image 2 to see if it's valid
    print("Stitching image")
    height1, width1, c = image1.shape
    height2, width2, c = image2.shape
    stitchheight = height1
    stitchwidth = width1

    # x is height, y is width
    x1, y1 = ransac.project(0, 0, homInv) # top left corner
    x2, y2 = ransac.project(width2, height2, homInv)  # upper right corner
    x3, y3 = ransac.project(width2, 0, homInv)
    x4, y4 = ransac.project(0, height2, homInv)

    minheight = min(y1, y3)
    maxheight = max(y2, y4)
    maxwidth = max(x2, x3)

    if minheight < 0:
        stitchheight -= minheight
        minheight = -minheight
    if maxheight > width1:
        stitchheight += (maxheight - stitchheight)
    if maxwidth > height1:
        stitchwidth = maxwidth


    stichedImage = np.zeros(shape=[stitchheight+5,stitchwidth+5,  3], dtype=np.uint8)
    stichedImage[minheight:height1+minheight, 0:width1] = image1
    cv2.imwrite('results_images/5.png', stichedImage)


    for x in range(stitchwidth):
        for y in range(stitchheight):
            newx, newy = ransac.project(x, y-minheight, hom)
            # need to check if coordinate is in image2 range
            if(newx<=width2 and newy<=height2 and newx>=0 and newy>=0):
                val = cv2.getRectSubPix(image2, (1, 1), (newx, newy))
                # if(stichedImage[y, x]!= 0):
                #     val = stichedImage[y, x](1-0.5) + val(0.5)
                stichedImage[y, x] = val
                print(stichedImage[y, x])

    cv2.imwrite('results_images/5.png', stichedImage)


    # # for each value in the second image project onto blank_image
    # for x in range(width2):
    #     for y in range(height2):
    #         newx, newy = ransac.project(x, y, homInv)
    #
    #         stichedImage[newy+minheight, newx] = image2[y, x]
    # cv2.imwrite('results_images/4.png', stichedImage)


