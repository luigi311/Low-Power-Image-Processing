import numpy as np

from dehaze.darktables.darktables import dehaze_darktables


def dehaze_images(numpy_images, method):
    """
    Dehaze a list of images.

    Parameters:
    numpy_images (np.ndarray): The images to dehaze.
    method (str): The dehazing method to use.

    Returns:
    np.ndarray: The dehazed images.

    """
    dehazed_images = []
    for image in numpy_images:
        # Apply the dehazing method to each image
        if method == "darktables":
            dehazed_image = dehaze_darktables(image)
        else:
            raise Exception("ERROR: Unknown dehazing method")
        dehazed_images.append(dehazed_image)
    return np.array(dehazed_images)
