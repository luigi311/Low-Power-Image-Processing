import numpy as np
import cv2


def filter_kernel(numpy_array):
    # Source https://www.analyticsvidhya.com/blog/2021/08/sharpening-an-image-using-opencv-library-in-python/
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    return cv2.filter2D(src=numpy_array, ddepth=-1, kernel=kernel)


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # Source https://stackoverflow.com/a/55590133
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def sharpen(numpy_array, method="filter_kernel", amount=1.0):
    try:
        if method == "filter_kernel":
            return filter_kernel(numpy_array)
        elif method == "unsharp_mask":
            return unsharp_mask(numpy_array, amount=amount)
        else:
            raise Exception(f"Sharpen Error: Sharpen method {method} not supported")
    except Exception as e:
        raise Exception(f"Sharpen Error: {e}")
