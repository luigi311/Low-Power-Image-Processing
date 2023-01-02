import rawpy, cv2, os, h5py
import numpy as np
from skimage.exposure import is_low_contrast
from requests import get
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def process_raw(dng_file):
    with rawpy.imread(dng_file) as raw:
        rgb = raw.postprocess(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            use_auto_wb=True,
            half_size=False,
            no_auto_bright=True,
            auto_bright_thr=0.01,
            no_auto_scale=False,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            gamma=(1, 1),
            highlight_mode=rawpy.HighlightMode(2),
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),
        )

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        return rgb


def save_hdf5(numpy_array, path):
    print("Saving hdf5 file")

    with h5py.File(f"{path}/images.hdf5", "w") as hdf5:
        hdf5.create_dataset(
            "images", np.shape(numpy_array), h5py.h5t.STD_U8BE, data=numpy_array
        )


def loadImages(path):
    """
    Load all dng, tiff or hdf5 images from a directory into a numpy array.

    Parameters:
    path (str): The path to the directory containing the images.

    Returns:
    np.ndarray: A numpy array containing all the images in the directory.

    """
    # Check if the path exists and is a directory
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError(f"ERROR {path} is not a valid directory.")

    # Remove trailing slash from path if present
    if path.endswith("/"):
        path = path[:-1]

    # Get a list of all files in the directory
    file_list = os.listdir(path)

    # Filter the list to only include dng, tiff, and hdf5 files
    process_file_list = [
        os.path.join(path, x) for x in file_list if x.endswith(("dng", "tiff", "hdf5"))
    ]

    # Preallocate the numpy array with the same dtype as the first image in the file list
    numpy_array = []

    # Check if there are any hdf5 files in the filtered list
    hdf5_files = [x for x in process_file_list if x.endswith("hdf5")]
    if hdf5_files:
        print("Loading hdf5 files")
        for hdf5_file in hdf5_files:
            # Load the images from the hdf5 file into the numpy array
            with h5py.File(hdf5_file, "r") as hdf5:
                numpy_array.append(np.array(hdf5["/images"][:]).astype(np.uint8))

        # Concatenate the images in numpy_array along the first axis
        numpy_array = np.concatenate(numpy_array, axis=0)
    else:
        dng_files = [x for x in process_file_list if x.endswith("dng")]
        tiff_files = [x for x in process_file_list if x.endswith("tiff")]
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            if dng_files:
                for result in executor.map(process_raw, dng_files):
                    numpy_array.append(result)
            if tiff_files:
                for result in executor.map(cv2.imread, tiff_files):
                    numpy_array.append(result)

    return np.array(numpy_array)


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
    filtered_array = np.empty(numpy_array.shape, dtype=numpy_array.dtype)

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
            filtered_array[i] = image

    # If no images passed the low contrast filter, return a copy of the input array with the first image removed
    if filtered_array.size == 0:
        print("All images low contrast, skipping first image only")
        return numpy_array[1:]

    # Return the filtered array
    return filtered_array[: i + 1]


def fixContrast(image):
    # get darkest and brightest pixel
    min_pixel = np.min(image)
    max_pixel = np.max(image)

    # Subtract darkest pixel from all pixels
    image = image - min_pixel

    # multiply to bring white to max_pixel
    image = image * (max_pixel / 255)

    return image


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


def downloader(url, file_name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


def future_thread_executor(args: list, workers: int = -1):
    futures_list = []
    results = []

    if workers == -1:
        workers = os.cpu_count() - 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for arg in args:
            # * arg unpacks the list into actual arguments
            futures_list.append(executor.submit(*arg))

        for future in futures_list:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise Exception(e)

    return results
