import argparse, os, cv2, traceback
import numpy as np

from time import time

from utils.utils import loadImages, filterLowContrast, save_hdf5, shrink_images


def save_image(path, image, extension="png"):
    if extension == "jpg":
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif extension == "png":
        cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    else:
        cv2.imwrite(path, image)


# Create main and do any processing if needed
def single_image(images, input_dir, histogram_method, image_extension="png", clip_limit=1.2, tile_grid_size=(8, 8)):
    # Default to second image if exists if not first
    image = images[1] if len(images) > 1 else images[0]

    if histogram_method != "none":
        image = single_histogram_processing(image, histogram_method, clip_limit, tile_grid_size)

    output_image = os.path.join(input_dir, f"main.{image_extension}")

    save_image(output_image, image, image_extension)
    print(f"Saved {output_image}")


def single_histogram_processing(image, histogram_method, clip_limit=1.2, tile_grid_size=(8, 8)):
    """
    Equalize the histogram of a single image.

    Parameters:
    image (np.ndarray): The image to process.
    histogram_method (str): The method to use for histogram enhancement.

    Returns:
    np.ndarray: The processed image.

    """
    # Check if the image is a grayscale image or has multiple channels
    if image.ndim == 2:
        # If the image is grayscale, apply the histogram enhancement method directly
        if histogram_method == "histogram_clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            image = clahe.apply(image)
        elif histogram_method == "histogram_equalize":
            image = cv2.equalizeHist(image)
        else:
            raise Exception("ERROR: Unknown histogram method")
    else:
        # If the image has multiple channels, convert it to the YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # Apply the histogram enhancement method to the Y channel
        if histogram_method == "histogram_clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])
        elif histogram_method == "histogram_equalize":
            yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        else:
            raise Exception("ERROR: Unknown histogram method")
        # Convert the image back to the RGB color space
        image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

    return image


def histogram_processing(numpy_array, histogram_method, clip_limit=1.2, tile_grid_size=(8, 8)):
    """
    Equalize the histograms of the images in a numpy array.

    Parameters:
    numpy_array (np.ndarray): A numpy array containing the images to process.
    histogram_method (str): The method to use for histogram enhancement.

    Returns:
    np.ndarray: A numpy array containing the processed images.

    """
    print("Histogram equalizing images")

    # Create a new numpy array to store the processed images
    processed_array = np.empty(numpy_array.shape, dtype=numpy_array.dtype)

    # Iterate over the images in the numpy array
    for i, image in enumerate(numpy_array):
        # Process the image
        processed_array[i] = single_histogram_processing(image, histogram_method, clip_limit, tile_grid_size)

    return processed_array


def setup_args():
    parser = argparse.ArgumentParser(description="Process Image")
    parser.add_argument("input_dir", help="Input directory of images")
    parser.add_argument(
        "--interal_image_extension",
        default="png",
        help="Extension of images to process",
    )
    parser.add_argument("--single_image", help="Single image mode", action="store_true")
    parser.add_argument(
        "--histogram_method",
        default="none",
        help="histogram method to use",
        choices=["histogram_clahe", "histogram_equalize"],
    )
    parser.add_argument(
        "--clip_limit",
        default=1.2,
        type=float,
        help="Clip limit for histogram_clahe",
    )
    parser.add_argument(
        "--tile_grid_size",
        default=8,
        type=int,
        help="Tile grid size for histogram_clahe",
    )
    parser.add_argument(
        "--dehaze_method",
        default="none",
        help="Dehaze method to use on all images",
        choices=["none", "darktables"],
    )
    parser.add_argument(
        "--color_method",
        default="none",
        help="Color method to use on final images",
        choices=["none", "image_adaptive_3dlut"],
    )
    parser.add_argument("--auto_stack", help="Auto stack images", action="store_true")
    parser.add_argument(
        "--stack_amount",
        default=3,
        type=int,
        help="Amount of images to stack at a time",
    )
    parser.add_argument(
        "--stack_method",
        help="Stacking method ORB (faster) or ECC (more precise)",
        choices=["ORB", "ECC"],
        default="ECC",
    )
    parser.add_argument("--show", help="Show result image", action="store_true")
    parser.add_argument(
        "--denoise_all",
        help="Denoise all images prior to stacking",
        action="store_true",
    )
    parser.add_argument(
        "--denoise_all_method",
        help="Denoise method for all images prior to stacking",
        choices=["fast", "fddnet", "ircnn"],
        default="fast",
    )
    parser.add_argument(
        "--denoise_all_amount",
        help="Denoise amount for all images prior to stacking",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--denoise",
        help="Denoise image",
        action="store_true",
    )
    parser.add_argument(
        "--denoise_method",
        help="Denoise image",
        choices=["fast", "fddnet", "ircnn", "none"],
        default="fast",
    )
    parser.add_argument("--denoise_amount", help="Denoise amount", type=int, default=2)
    parser.add_argument(
        "--super_resolution", help="Super resolution", action="store_true"
    )
    parser.add_argument(
        "--super_resolution_method",
        help="Super Resolution method",
        choices=["ESPCN", "FSRCNN"],
        default="ESPCN",
    )
    parser.add_argument(
        "--super_resolution_scale",
        help="Super Resolution Scale",
        choices=[2, 4],
        type=int,
        default=2,
    )
    parser.add_argument("--shrink_images", help="Shrink image", action="store_true")
    parser.add_argument(
        "--scale_down",
        type=int,
        default=720,
        help="Scale down image to the following resolution for stacking and filter contrast",
    )
    parser.add_argument(
        "--parallel_raw",
        type=int,
        default=None,
        help="Number of parallel pyraw processes to use",
    )
    parser.add_argument(
        "--sharpen",
        help="Sharpen the postprocess image",
        choices=["filter_kernel", "unsharp_mask"],
        type=str
    )

    return parser.parse_args()


# ===== MAIN =====
def main(args):
    total_tic = time()

    # Flag to indicate if any processing was done on the image
    processed_image = False

    loading_tic = time()
    image_folder = args.input_dir

    # Load all images
    numpy_images = loadImages(image_folder, args.parallel_raw)

    # if image_folder/images.hdf5 does not exists create hdf5 file containing filtered images
    if not os.path.isfile(os.path.join(image_folder, "images.hdf5")):
        # Filter low contrast images
        numpy_images = filterLowContrast(numpy_images, args.scale_down)

        # Save filtered images to hdf5
        save_hdf5(numpy_images, image_folder)

    print(f"Loaded {len(numpy_images)} images in {time() - loading_tic} seconds")

    if args.single_image:
        # Create main image
        main_tic = time()
        print("Creating main image")

        single_image(
            numpy_images,
            args.input_dir,
            args.histogram_method,
            args.interal_image_extension,
            args.clip_limit,
            (args.tile_grid_size, args.tile_grid_size),
        )

        print(f"Created main image in {time() - main_tic} seconds")

        print(f"Total {time() - total_tic} seconds")

        # Exit if single_image is ran to avoid processing other images
        exit(0)

    if args.shrink_images:
        shrink_tic = time()

        numpy_images = shrink_images(numpy_images)

        processed_image = True
        print(f"Shunk in {time() - shrink_tic} seconds")

    if args.histogram_method != "none":
        equalize_tic = time()

        numpy_images = histogram_processing(numpy_images, args.histogram_method, args.clip_limit, (args.tile_grid_size, args.tile_grid_size))

        processed_image = True
        print(f"Histogram equalized in {time() - equalize_tic} seconds")

    if args.dehaze_method != "none":
        dehaze_tic = time()

        print("Dehazing images")
        dehazed_images = []

        from dehaze.dehaze import dehaze_images

        numpy_images = dehaze_images(numpy_images, args.dehaze_method)

        processed_image = True
        print(f"Dehazed {len(dehazed_images)} images in {time() - dehaze_tic} seconds")

    if args.denoise_all:
        try:
            denoise_all_tic = time()

            from denoise.denoise import denoise_images

            numpy_images = denoise_images(
                numpy_images,
                method=args.denoise_all_method,
                amount=args.denoise_all_amount,
            )

            processed_image = True
            print(f"Denoised all images in {time() - denoise_all_tic} seconds")

        except Exception as e:
            raise Exception(f"ERROR: Could not denoise all images\n{e}")

    if args.auto_stack:
        try:
            stack_tic = time()

            from stacking.stacking import stacker

            image = stacker(
                numpy_images, args.stack_amount, args.stack_method, args.scale_down
            )

            processed_image = True
            print(f"Stacked images in {time() - stack_tic} seconds")

        except Exception as e:
            raise Exception(f"ERROR: Could not stack images\n{e}")
    else:
        image = numpy_images[0]

    if args.sharpen:
        from sharpen.sharpen import sharpen

        print("Sharpen")
        sharp_tic = time()
        image = sharpen(image, args.sharpen)

        processed_image = True
        print(f"Sharpen image in {time() - sharp_tic} seconds")

    if args.denoise:
        try:
            denoise_tic = time()

            from denoise.denoise import denoiser

            denoiser(image, args.denoise_method, args.denoise_amount)

            processed_image = True
            print(f"Denoised image in {time() - denoise_tic} seconds")

        except Exception as e:
            raise Exception(f"ERROR: Could not denoise image\n{e}")

    if args.color_method != "none":
        try:
            color_tic = time()
            print("Color adjusting image")

            if args.color_method == "image_adaptive_3dlut":
                from color.image_adaptive_3dlut.image_adaptive_3dlut import (
                    image_adaptive_3dlut,
                )

                image = image_adaptive_3dlut(image, "sRGB")

            processed_image = True
            print(f"Color adjusted image in {time() - color_tic} seconds")
        except Exception as e:
            raise Exception(f"Error: Failed to adjuts color\n{e}")

    if args.super_resolution:
        from super_resolution.super_resolution import super_resolution

        print("Super resolution")
        super_tic = time()
        image = super_resolution(
            image, args.super_resolution_method, args.super_resolution_scale
        )

        processed_image = True
        print(f"Super resolution image in {time() - super_tic} seconds")

    if args.shrink_images:
        from super_resolution.super_resolution import super_resolution

        print("Resize shrunk image")
        super_tic = time()
        image = super_resolution(image, args.super_resolution_method, 2)

        processed_image = True
        print(f"Super resolution image in {time() - super_tic} seconds")

    if processed_image:
        process_tic = time()
        print("Save process image")

        output_image = os.path.join(
            args.input_dir, f"main_processed.{args.interal_image_extension}"
        )
        save_image(output_image, image, args.interal_image_extension)

        print(f"Saved {output_image} in {time() - process_tic}")

    print(f"Total {time() - total_tic} seconds")


if __name__ == "__main__":
    try:
        args = setup_args()
        main(args)
    except Exception as error:
        if isinstance(error, list):
            for message in error:
                print(message)
        else:
            print(error)

        print(traceback.format_exc())
