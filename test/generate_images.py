import random, os, sys, cv2
import numpy as np

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from utils.utils import loadImages


def noisy(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    lum = yuv_image[:, :, 0]

    # Add salt and pepper noise to images
    row, col = lum.shape

    number_of_pixels = int(row * col * 0.1)
    for _ in range(number_of_pixels):
        x = random.randint(0, row - 1)
        y = random.randint(0, col - 1)
        if random.random() < 0.5:
            lum[x, y] = 0
        else:
            lum[x, y] = 255

    yuv_image[:, :, 0] = lum
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    return rgb_image


def generate_noise():
    numpyimages = loadImages("test/images")

    # Generate 3 noisy images
    noisy_images = []
    for i in range(3):
        noisy_images.append(noisy(numpyimages[0]))

    noisy_images = np.array(noisy_images)

    # Create noisy_images folder
    if not os.path.exists("test/noisy_images"):
        os.makedirs("test/noisy_images")

    # Save noisy images
    for i, image in enumerate(noisy_images):
        cv2.imwrite(f"test/noisy_images/{i}.tiff", image)



generate_noise()
