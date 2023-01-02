import numpy as np
import cv2


def alignImageECC(
    imageF,
    shrunk_image,
    first_image_shrunk,
    shrink_factor,
    warp_matrix,
    warp_mode,
    criteria,
    h,
    w,
):
    try:
        # Estimate perspective transform
        _, warp_matrix = cv2.findTransformECC(
            first_image_shrunk,
            cv2.cvtColor(shrunk_image, cv2.COLOR_RGB2GRAY),
            warp_matrix,
            warp_mode,
            criteria,
        )

        # Adjust the warp_matrix to the scale of the original images
        warp_matrix = (
            warp_matrix
            * np.array(
                [
                    [1, 1, 1 / shrink_factor],
                    [1, 1, 1 / shrink_factor],
                    [shrink_factor, shrink_factor, 1],
                ]
            )
        ).astype(np.float32)

        # Align image to first image
        image_align = cv2.warpPerspective(
            imageF,
            warp_matrix,
            (h, w),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

        return image_align
    except:
        return None


def stackImagesECCWorker(numpy_array, scale_down=720):
    """
    Align and stack images with the ECC (Extended Correlation Coefficient) method.
    This method is slower but more accurate than other methods such as SIFT or SURF.

    Parameters:
    numpy_array (np.ndarray): A numpy array of images. All images must be the same size and type.
    scale_down (int): The scale to which the images will be resized before alignment. The smallest
        side of the image will be resized to this value. (default=720)

    Returns:
    np.ndarray: A stacked image of the input images, aligned using the ECC method.

    """

    # Check if input is a valid numpy array of images
    if not isinstance(numpy_array, np.ndarray) or numpy_array.ndim != 4:
        raise ValueError("Input must be a numpy array of images.")

    # Check if all images are the same size
    sizes = {tuple(img.shape[:2]) for img in numpy_array}
    if len(sizes) > 1:
        raise ValueError("All images must be the same size.")

    # Return an empty image if the input array is empty
    if len(numpy_array) == 0:
        return np.zeros(sizes.pop(), dtype=numpy_array.dtype)

    # Return the first image if the input array has only one element
    if len(numpy_array) == 1:
        return numpy_array[0]

    # Set the warp mode to homography
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    w, h = numpy_array[0].shape[:2]

    # Shrink_factor to bring the image to scale_down on the smallest side
    shrink_factor = scale_down / min(w, h)

    # Specify the number of iterations.
    number_of_iterations = 5

    # Specify the threshold of the increment in the correlation coefficient
    # between two iterations
    termination_eps = 1e-10

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    first_image = None
    first_image_shrunk = None
    stacked_image = None
    count_stacked = 1

    for _, image in enumerate(numpy_array):
        imageF = image.astype(np.float32) / 255
        shrunk_image = cv2.resize(image, (0, 0), fx=shrink_factor, fy=shrink_factor)
        # Convert to gray scale floating point image
        if first_image is None:
            first_image = cv2.cvtColor(imageF, cv2.COLOR_RGB2GRAY)
            first_image_shrunk = cv2.cvtColor(shrunk_image, cv2.COLOR_RGB2GRAY)
            stacked_image = imageF
        else:
            image_align = alignImageECC(
                imageF,
                shrunk_image,
                first_image_shrunk,
                shrink_factor,
                warp_matrix,
                warp_mode,
                criteria,
                h,
                w,
            )
            if image_align is None:
                print("Failed to align image")
            else:
                stacked_image += image_align
                count_stacked += 1
                print("Aligned image")

    stacked_image = (stacked_image / count_stacked) * 255
    stacked_image = stacked_image.astype(np.uint8)

    return stacked_image


def chunker(numpy_array, method="ECC", stacking_amount=3, scale_down=720):
    """
    Stack a series of images using the ECC (Extended Correlation Coefficient) method.

    Parameters:
    numpy_array (np.ndarray): A numpy array of images. All images must be the same size and type.
    stacking_amount (int): The number of images to stack at a time. (default=3)
    scale_down (int): The scale to which the images will be resized before alignment. The smallest
        side of the image will be resized to this value. (default=720)

    Returns:
    np.ndarray: A stacked image of the input images, aligned using the ECC method.

    """
    # Split the input array into chunks of size stacking_amount
    chunks = [
        numpy_array[x : x + stacking_amount]
        for x in range(0, len(numpy_array), stacking_amount)
    ]

    stacked = []

    # Stack each chunk using the ECC method
    for chunk in chunks:
        if len(chunk) > 1:
            if method == "ECC":
                stacked.append(stackImagesECCWorker(chunk, scale_down))
            elif method == "ORB":
                stacked.append(stackImagesKeypointMatching(chunk))
        else:
            stacked.append(chunk[0])

    # While there are more than 1 image in the stacked array, keep stacking using the ECC method
    while len(stacked) > 1:
        temp_stacked = []
        # Split the stacked array into chunks of size stacking_amount
        chunks = [
            stacked[x : x + stacking_amount]
            for x in range(0, len(stacked), stacking_amount)
        ]

        # Stack each chunk using the ECC method
        for chunk in chunks:
            if len(chunk) > 1:
                if method == "ECC":
                    temp_stacked.append(
                        stackImagesECCWorker(np.array(chunk), scale_down)
                    )
                elif method == "ORB":
                    temp_stacked.append(stackImagesKeypointMatching(np.array(chunk)))
            else:
                temp_stacked.append(chunk[0])

        stacked = temp_stacked

    # Return the final stacked image
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


def stacker(numpy_array, stacking_amount=3, method="ECC", scale_down=720):
    try:
        if method in ["ECC", "ORB"]:
            return chunker(numpy_array, method, stacking_amount, scale_down)
        else:
            raise Exception(f"Stacking Error: Stacking method {method} not supported")
    except Exception as e:
        raise Exception(f"Stacking Error: {e}")
