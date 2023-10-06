import rawpy, cv2, exifread, os
import numpy as np

from math import floor
from skimage.exposure import is_low_contrast
from concurrent.futures import ProcessPoolExecutor

from utils.utils import files


def process_raw(dng_file, half_size=False, auto_white_balance=False):
    raw = rawpy.imread(dng_file)

    raw_params = {
        "demosaic_algorithm": rawpy.DemosaicAlgorithm.AHD,
        "use_auto_wb": auto_white_balance,
        "half_size": half_size,
        "no_auto_bright": True,
        "auto_bright_thr": 0.01,
        "no_auto_scale": False,
        "output_color": rawpy.ColorSpace.sRGB,
        "output_bps": 8,
        "gamma": (2.222, 4.5),
        "highlight_mode": rawpy.HighlightMode(2),
        "fbdd_noise_reduction": rawpy.FBDDNoiseReductionMode(0),
    }

    image = raw.postprocess(**raw_params)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def loadImages(path, threads=None, half_size=False, auto_white_balance=False):
    """
    Load all dng, tiff or hdf5 images from a directory into a numpy array.

    Parameters:
    path (str): The path to the directory containing the images.

    Returns:
    np.ndarray: A numpy array containing all the images in the directory.

    """

    dng_files, tiff_files = files(path)

    images = []

    # Default to half the number of cpu cores due to rawpy using multiple threads
    workers = threads if threads else max(floor(os.cpu_count() / 2), 1)
    # Instead of appending, concatenate the result images to numpy_array
    with ProcessPoolExecutor(max_workers=workers) as executor:
        if dng_files:
            for result_image in executor.map(
                process_raw,
                dng_files,
                [half_size] * len(dng_files),
                [auto_white_balance] * len(dng_files),
            ):
                images.append(result_image)

        if tiff_files:
            for result_image in executor.map(cv2.imread, tiff_files):
                images.append(result_image)

    return np.array(images)


def generate_exif(path: str):
    tiff_files, dng_files = files(path)

    image = tiff_files[-1] if tiff_files else dng_files[-1]

    with open(image, "rb") as image_file:
        tags = exifread.process_file(image_file, details=False)

    # Convert all tags to strings
    for key, value in tags.items():
        tags[key] = str(value)

    # Save exif data to a file
    with open(os.path.join(path, "exif.txt"), "w") as exif_file:
        exif_file.write(str(tags))


def save_image(path, image, extension="png", quality=95):
    if extension == "jpg":
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif extension == "png":
        cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    else:
        cv2.imwrite(path, image)


def filterLowContrast(numpy_array, scale_down=720):
    """
    Filter out images with low contrast from a numpy array of images.

    Parameters:
    numpy_array (np.ndarray): A numpy array of images. All images must be the same size and type.
    scale_down (int): The scale to which the images will be resized before filtering. The smallest
        side of the image will be resized to this value. (default=720)

    Returns:
    np.ndarray: A numpy array containing only the images that passed the low contrast filter.

    """
    # Check if input is a valid numpy array of images
    if not isinstance(numpy_array, np.ndarray) or numpy_array.ndim != 4:
        raise ValueError("Input must be a numpy array of images.")

    # Check if all images are the same size
    sizes = {tuple(img.shape[:2]) for img in numpy_array}
    if len(sizes) > 1:
        raise ValueError("All images must be the same size.")

    w, h = numpy_array[0].shape[:2]
    shrink_factor = scale_down / min(w, h)

    # Preallocate the filtered_array with the same shape and dtype as the input array
    filtered_array = []
    # Iterate over the images in the input array
    for i, image in enumerate(numpy_array):
        # Check if the image is low contrast
        if not is_low_contrast(
            cv2.resize(image, (0, 0), fx=shrink_factor, fy=shrink_factor),
            fraction_threshold=0.05,
            lower_percentile=10,
            upper_percentile=90,
            method="linear",
        ):
            # If not low contrast, append to the filtered_array
            filtered_array.append(image)
        else:
            # If low contrast, print the image number and continue
            print(f"Image {i} low contrast, skipping")

    filtered_array = np.array(filtered_array)

    # If less than 2 images passed the low contrast filter, return a copy of the input array with the first image removed
    if len(filtered_array) < 2:
        print("Less than 2 images with good contrast, skipping first image only")
        return numpy_array[1:]

    # Return the filtered array
    return filtered_array


def shrink_images(numpy_array):
    """
    Shrink the images in a numpy array to half their size.

    Parameters:
    numpy_array (np.ndarray): A numpy array containing the images to shrink.

    Returns:
    np.ndarray: A numpy array containing the shrunken images.

    """
    print("Shrinking images")

    # Calculate the new dimensions of the images
    new_height = numpy_array.shape[1] // 2
    new_width = numpy_array.shape[2] // 2

    # Check if either dimension is 0
    if new_height == 0 or new_width == 0:
        raise ValueError("One or both dimensions is 0. Cannot shrink images.")

    # Create a new numpy array to store the shrunken images
    shrunken_array = np.empty(
        (numpy_array.shape[0], new_height, new_width, 3), dtype=numpy_array.dtype
    )

    # Iterate over the images in the numpy array
    for i, image in enumerate(numpy_array):
        # Resize the image to half its size
        shrunken_array[i] = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    return shrunken_array
