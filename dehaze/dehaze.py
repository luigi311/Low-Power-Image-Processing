import numpy as np

from dehaze.darktables.darktables import dehaze_darktables

def dehaze_image(image, method):
    """
    Dehaze a single image.

    Parameters:
    image (np.ndarray): The image to dehaze.
    method (str): The dehazing method to use.

    Returns:
    np.ndarray: The dehazed images.

    """
    if method == "darktables":
        out_image = dehaze_darktables(image)
    else:
        raise Exception("ERROR: Unknown dehazing method")
    
    return out_image

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
        out_image = dehaze_image(image, method)
        dehazed_images.append(out_image)
    
    return np.array(dehazed_images)
