import numpy as np
import cv2

# Align and stack images with ECC method
# Slower but more accurate

def stackImagesECCWorker(numpy_array):
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None

    for _, image in enumerate(numpy_array):
        # image = cv2.imread(file, 1).astype(np.float32) / 255
        imageF = image.astype(np.float32) / 255
        if first_image is None:
            # convert to gray scale floating point image
            # first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # stacked_image = image
            first_image = cv2.cvtColor(imageF, cv2.COLOR_BGR2GRAY)
            stacked_image = imageF
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(
                cv2.cvtColor(imageF, cv2.COLOR_BGR2GRAY),
                first_image,
                M,
                cv2.MOTION_HOMOGRAPHY,
            )
            w, h, _ = imageF.shape
            # Align image to first image
            image_align = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += image_align

    stacked_image /= len(numpy_array)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image

def stackImagesECC(numpy_array, stacking_amount=3):
    if stacking_amount == 1:
        print("Error: Stacking amount must be greater than 1")
        exit(1)
    
    stacked = []
    # split into chunks of size stacking_amount
    chunks = [numpy_array[x:x + stacking_amount] for x in range(0, len(numpy_array), stacking_amount)]

    print(len(chunks))
    for chunk in chunks:
        if len(chunk) > 1:
            stacked.append(stackImagesECCWorker(chunk))
        else:
            stacked.append(chunk[0])

    #recursively stack images into there is only one image left
    while len(stacked) > 1:
        stacked = [stackImagesECC(stacked, stacking_amount)]

    if len(stacked) == 1:
        return stacked[0]
        
# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(numpy_array):

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for _, image in enumerate(numpy_array):
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32([first_kp[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(numpy_array)
    stacked_image = (stacked_image * 255).astype(np.uint8)
    return stacked_image
