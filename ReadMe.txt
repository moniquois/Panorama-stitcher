Readme.txt

Run main.py to run the program. All the images from the program are stored in the results_images
folder as named in the project description. My own harris corner and keypoint detector are used for part 1&2. In the main file
you'll see a function called completePanorama() commented out at the end which will stitch together the entire Rainier panorama

Part :  Compute Harris Corner detection
In the file HarrisCorner, the function computeHarris() is used to find the key points. fdkeypoints() calls the computeHarris function
saves the results of the computeHarris() function.

Part 2: Matching the interest points
In the file, featuredetect, the function SIFT() is used to create descriptors for the key points using the openCV sift function.
The match() function matches the features using the ratio test and saves the results; returning the best matches between the two images
The file also contains the self implemented SIFT function createfeatures() which creates features using featuredescript(),
hist() and matchfeatures() is used to match the features. This is only used for part 2 and not the rest of the project.

Part 3: Compute homography
In the file, ransac, there is a function called RANSAC() which finds the best homography/ inverse homography for the two images.
project() function which projects a point given a homography,
computeInlierCount() which finds the number of inliers given a homography,
findInliers() which returns the matches that are inliers given a homography,
and keys() function which finds the keypoints of the two images given a Dmatch object

Part 4: Stitch the images
In the file, stitched,
resizeimage() function finds the size of the image needed to stitch two images together by projecting the corners of the image,
stitch() Copy image1 into new image at proper position, project each point in new image if point lies within image2
boundaries add pixel value

Bonus:
Stitching togther all Rainier image by using a function called panorama() in the main file which loops to add
each new rainier image to the stitchedimage.