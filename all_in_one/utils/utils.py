import rawpy, cv2, os
import numpy as np
from skimage.exposure import is_low_contrast
from requests import get


def process_raw(dng_file):
    with rawpy.imread(dng_file) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),
        )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb


# Create a numpy array for all the dng images in the folder
def createNumpyArray(path):
    file_list = os.listdir(path)
    dng_file_list = [os.path.join(path, x) for x in file_list if x.endswith(".dng")]

    # Read image to get dimensions
    second_image = process_raw(dng_file_list[1])
    h, w, _ = second_image.shape

    # Create numpy array
    numpy_array = np.zeros((len(dng_file_list), h, w, 3), dtype=np.uint8)

    # Read all images into numpy array
    for i, file in enumerate(dng_file_list):
        if i == 1:
            numpy_array[i] = second_image
        else:
            numpy_array[i] = process_raw(file)

    return numpy_array


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
    return filtered_array


def downloader(url, file_name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)
