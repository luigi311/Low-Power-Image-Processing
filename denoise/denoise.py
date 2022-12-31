import numpy as np


def denoise_images(numpy_images, method, amount):
    """
    Denoise a list of images.

    Parameters:
    numpy_images (np.ndarray): The images to denoise.
    method (str): The denoising method to use.
    amount (float): The amount of denoising to apply.

    Returns:
    np.ndarray: The denoised images.

    """
    denoised_images = []
    for image in numpy_images:
        denoised_image = denoiser(image, method, amount)
        denoised_images.append(denoised_image)

    return np.array(denoised_images)


def denoiser(image, method, amount):
    try:
        print(f"Denoising image via {method} with amount {amount}")

        if method == "fast":
            from denoise.opencv_denoise.opencv_denoise import fastDenoiseImage

            image = fastDenoiseImage(image, amount)

        elif method == "fddnet":
            from denoise.fddnet.fddnet import fddnetDenoiseImage

            image = fddnetDenoiseImage(image, amount)

        elif method == "ircnn":
            from denoise.ircnn.ircnn import ircnnDenoiseImage

            image = ircnnDenoiseImage(image, amount)

        else:
            raise Exception(f"ERROR: Denoise method {method} not found!")

        return image

    except Exception as e:
        raise Exception(e)
