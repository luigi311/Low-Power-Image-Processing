import numpy as np
import cv2

# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECCWorker(numpy_array):
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    first_image = None
    stacked_images = None

    first_image_shrunk = None

    w, h, _ = numpy_array[0].shape

    shrink_factor = 0.6

    if min(w,h) < 1080:
        shrink_factor = 1

    for _, image in enumerate(numpy_array):
        imageF = image.astype(np.float32) / 255
        shrunk_image = cv2.resize(image, (0, 0), fx=shrink_factor, fy=shrink_factor)
        shrunk_image = shrunk_image.astype(np.float32) / 255

        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(imageF, cv2.COLOR_RGB2GRAY)
            first_image_shrunk = cv2.cvtColor(shrunk_image, cv2.COLOR_RGB2GRAY)

            stacked_images = imageF
        else:
            # Estimate perspective transform
            _, warp_matrix = cv2.findTransformECC(
                first_image_shrunk,
                cv2.cvtColor(shrunk_image, cv2.COLOR_RGB2GRAY),
                warp_matrix,
                warp_mode,
                criteria,
            )

            # Align image to first image
            image_align = cv2.warpPerspective(imageF, warp_matrix, (h, w), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            stacked_images += image_align

    stacked_images /= len(numpy_array)

    stacked_image = (stacked_images * 255).astype(np.uint8)

    return stacked_image

def stackImagesECC(numpy_array, stacking_amount=3):
    if stacking_amount == 1:
        print("Error: Stacking amount must be greater than 1")
        exit(1)

    stacked = []
    # split into chunks of size stacking_amount
    chunks = [
        numpy_array[x : x + stacking_amount]
        for x in range(0, len(numpy_array), stacking_amount)
    ]

    for chunk in chunks:
        if len(chunk) > 1:
            stacked.append(stackImagesECCWorker(chunk))
        else:
            stacked.append(chunk[0])

    # recursively stack images into there is only one image left
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
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(numpy_array)
    stacked_image = (stacked_image * 255).astype(np.uint8)

    return stacked_image


def stacker(numpy_array, stacking_amount=3, method="ECC"):
    try:
        if method == "ECC":
            return stackImagesECC(numpy_array, stacking_amount)
        elif method == "ORB":
            return stackImagesKeypointMatching(numpy_array)
        else:
            raise Exception(f"Stacking Error: Stacking method {method} not supported")
    except Exception as e:
        raise Exception(f"Stacking Error: {e}")
