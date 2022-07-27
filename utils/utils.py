import rawpy, cv2, os
import numpy as np
from skimage.exposure import is_low_contrast
from requests import get
from concurrent.futures import ThreadPoolExecutor


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


# Create a numpy array for all the dng images in the folder
def loadImages(path):
    try:
        if not os.path.exists(path):
            raise Exception(f"loadImages: ERROR {path} not found!")

        if path.endswith("/"):
            path = path[:-1]

        extensions = tuple(["dng", "tiff"])
        file_list = os.listdir(path)
        process_file_list = [os.path.join(path, x) for x in file_list if x.endswith(extensions)]

        # Create numpy array
        numpy_array = []

        # Read all images into numpy array
        for file in process_file_list:
            if file.endswith("dng"):
                numpy_array.append(process_raw(file))
            else:
                numpy_array.append(cv2.imread(file))
                
        return numpy_array

    except Exception as e:
        raise Exception(e)


# Filter out images with low contrast from numpy array
def filterLowContrast(numpy_array):
    filtered_array = []
    for i, image in enumerate(numpy_array):
        if is_low_contrast(
            image,
            fraction_threshold=0.05,
            lower_percentile=10,
            upper_percentile=90,
            method="linear",
        ):
            print(f"Image {i} is low contrast")
        else:
            filtered_array.append(image)

    # if no filtered images, return original array expect for the first image
    if len(filtered_array) == 0:
        print("All images low contrast, skipping first image only")
        for i in range(1, len(numpy_array)):
            filtered_array.append(numpy_array[i])

    return filtered_array


def fixContrast(image):
    # get darkest and brightest pixel
    min_pixel = np.min(image)
    max_pixel = np.max(image)

    # Subtract darkest pixel from all pixels
    image = image - min_pixel

    # multiply to bring white to max_pixel
    image = image * (max_pixel / 255)

    return image


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
        workers = os.cpu_count()

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
