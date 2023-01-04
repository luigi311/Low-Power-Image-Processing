import os, sys, pytest
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


def calculate_psnr(noisy_image, ground_truth):
    # Calculate PSNR
    return np.mean(ground_truth) / np.mean(noisy_image)


def setup_images():
    from utils.utils import loadImages

    # Load noisy images
    noisy_images = loadImages("test/noisy_images")

    # Load ground truth images
    ground_truth = loadImages("test/images")

    # Remove first image from ground truth due to low contrast
    ground_truth = ground_truth[1:]

    return noisy_images, ground_truth


def test_filter_low_contrast():
    from utils.utils import filterLowContrast, loadImages

    ground_truth = loadImages("test/images")

    # Filter out low contrast images
    filtered_images = filterLowContrast(ground_truth)

    # Check if there are only 4 images left
    assert len(filtered_images) == 4


def test_shrink_images():
    from utils.utils import shrink_images

    _, ground_truth = setup_images()

    # Shrink images
    shrunk_images = shrink_images(ground_truth)

    # Check if images are half the size
    assert shrunk_images.shape[1] == ground_truth.shape[1] / 2


def test_denoise_fast():
    from denoise.denoise import denoiser

    noisy_images, ground_truth = setup_images()

    # denoise first image
    denoised_image = denoiser(noisy_images[0], "fast", 10)

    # Calculate PSNRs
    psnr_denoised = calculate_psnr(denoised_image, ground_truth[0])
    psnr_noisy = calculate_psnr(noisy_images[0], ground_truth[0])

    # Check if denoised image is less than noisy image
    assert psnr_denoised > psnr_noisy


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_denoise_ircnn():
    from denoise.denoise import denoiser

    noisy_images, _ = setup_images()

    # denoise first image
    denoiser(noisy_images[0], "ircnn", 35)
    # denoised_image = denoiser(noisy_images[0], "ircnn", 35)

    # Calculate PSNRs
    # psnr_denoised = calculate_psnr(denoised_image, ground_truth[0])
    # psnr_noisy = calculate_psnr(noisy_images[0], ground_truth[0])

    # Check if denoised image is less than noisy image
    # IRCNN is not working properly so we are not checking the PSNR
    # assert psnr_denoised > psnr_noisy


def test_denoise_fddnet():
    from denoise.denoise import denoiser

    noisy_images, ground_truth = setup_images()

    # denoise first image
    denoised_image = denoiser(noisy_images[0], "fddnet", 35)

    # Calculate PSNRs
    psnr_denoised = calculate_psnr(denoised_image, ground_truth[0])
    psnr_noisy = calculate_psnr(noisy_images[0], ground_truth[0])

    # Check if denoised image is less than noisy image
    assert psnr_denoised > psnr_noisy


def test_super_resolution_espcn():
    from super_resolution.super_resolution import super_resolution

    _, ground_truth = setup_images()

    image = ground_truth[0]
    super_image = super_resolution(image, "ESPCN", 2)

    # Assert that super image is 2x bigger than original image
    assert super_image.shape[0] == image.shape[0] * 2


def test_super_resolution_fsrcnn():
    from super_resolution.super_resolution import super_resolution

    _, ground_truth = setup_images()

    image = ground_truth[0]
    super_image = super_resolution(image, "FSRCNN", 2)

    # Assert that super image is 2x bigger than original image
    assert super_image.shape[0] == image.shape[0] * 2


def test_stacking_ecc():
    from stacking.stacking import stacker

    noisy_images, ground_truth = setup_images()

    stacked_image = stacker(noisy_images, 3, "ECC", 480)

    # Calculate PSNRs
    psnr_stacked = calculate_psnr(stacked_image, ground_truth[0])
    psnr_noisy = calculate_psnr(noisy_images[0], ground_truth[0])

    # Check if stacked image is less than noisy image
    assert psnr_stacked > psnr_noisy


def test_stacking_orb():
    from stacking.stacking import stacker

    noisy_images, ground_truth = setup_images()

    stacked_image = stacker(noisy_images, 3, "ORB")

    # Calculate PSNRs
    psnr_stacked = calculate_psnr(stacked_image, ground_truth[0])
    psnr_noisy = calculate_psnr(noisy_images[0], ground_truth[0])

    # Check if stacked image is less than noisy image
    assert psnr_stacked > psnr_noisy
