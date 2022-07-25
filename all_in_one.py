import argparse, os, cv2

import numpy
from time import time

from utils.utils import loadImages, filterLowContrast

# Create main and do any processing if needed
def single_image(images, input_dir, image_extension="png"):
    # Default to second image if exists if not first
    if len(images) > 1:
        image = images[1]
    else:
        image = images[0]

    output_image = os.path.join(input_dir, f"main.{image_extension}")
    print(f"Saved {output_image}")
    cv2.imwrite(output_image, image)

    return image


def histogram_processing(image, contrast_method):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    if contrast_method == "histogram_clahe":
        # equalize with clahe
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])

    elif contrast_method == "histogram_equalize":
        # equalize with equalizeHist
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

    else:
        print("ERROR: Unknown contrast method")
        exit(1)

    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)


def shrink_images(numpy_array):
    print("Shrinking images")
    # Shrink images to half size
    for i, image in enumerate(numpy_array):
        numpy_array[i] = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    return numpy_array

# ===== MAIN =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Image")
    parser.add_argument("input_dir", help="Input directory of images")
    parser.add_argument(
        "--interal_image_extension",
        default="png",
        help="Extension of images to process",
    )
    parser.add_argument("--single_image", help="Single image mode", action="store_true")
    parser.add_argument(
        "--contrast_method",
        default="none",
        help="Contrast method to use",
        choices=["histogram_clahe", "histogram_equalize"],
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
        "--denoise_method",
        help="Denoise image",
        choices=["fast", "fddnet", "ircnn", "none"],
        default="none",
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
    parser.add_argument(
        "--shrink_images", help="Shrink image", action="store_true"
    )
    args = parser.parse_args()

    processed_image = False
    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print(f"ERROR {image_folder} not found!")
        exit()

    if image_folder.endswith("/"):
        image_folder = image_folder[:-1]

    loading_tic = time()
    numpy_images = loadImages(image_folder)

    print("Filtering low contrast images")
    numpy_images = filterLowContrast(numpy_images)

    print(f"Loaded {len(numpy_images)} images in {time()-loading_tic} seconds")

    if args.single_image:
        # Create main image
        print("Creating main image")
        main_tic = time()

        image = single_image(numpy_images, args.input_dir, args.interal_image_extension)
        if args.show:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Created main image in {time()-main_tic} seconds")

    if args.shrink_images:
        numpy_images = shrink_images(numpy_images)

    if args.contrast_method != "none":
        equalized_images = []

        print("Histogram equalizing images")
        equalize_tic = time()

        for image in numpy_images:
            equalized_images.append(histogram_processing(image, args.contrast_method))

        numpy_images = equalized_images
        print(f"Histogram equalized in {time()-equalize_tic} seconds")

    if args.dehaze_method != "none":
        dehaze_tic = time()

        print("Dehazing images")
        dehazed_images = []

        if args.dehaze_method == "darktables":
            from dehaze.darktables.darktables import dehaze_darktables

            for image in numpy_images:
                dehazed_images.append(dehaze_darktables(image))

        print(f"Dehazed {len(dehazed_images)} images in {time()-dehaze_tic} seconds")
        numpy_images = dehazed_images

    if args.denoise_all:
        try:
            denoise_all_tic = time()

            numpy_images_denoised = []

            from denoise.denoise import denoiser

            for image in numpy_images:
                numpy_images_denoised.append(
                    denoiser(image, method=args.denoise_all_method, amount=args.denoise_all_amount)
                )

            numpy_images = numpy_images_denoised
            print(f"Denoised all images in {time()-denoise_all_tic} seconds")

        except Exception as e:
            print("ERROR: Could not denoise all images\n", e)

    if args.auto_stack:
        try:
            stack_tic = time()

            from stacking.stacking import stacker
            image = stacker(numpy_images, args.stack_amount, args.stack_method)

            processed_image = True
            print(f"Stacked images in {time() - stack_tic} seconds")

        except Exception as e:
            print(f"ERROR: Could not stack images {e}")


    if args.denoise_method != "none":
        try:
            denoise_tic = time()

            from denoise.denoise import denoiser
            denoiser(image, args.denoise_method, args.denoise_amount)

            processed_image = True
            print(f"Denoised image in {time()-denoise_tic} seconds")

        except Exception as e:
            print("ERROR: Could not denoise image\n", e)

    if args.color_method != "none":
        color_tic = time()
        print("Color adjusting image")

        if args.color_method == "image_adaptive_3dlut":
            from color.image_adaptive_3dlut.image_adaptive_3dlut import (
                image_adaptive_3dlut,
            )

            image = image_adaptive_3dlut(image, "sRGB")

        else:
            print(f"ERROR: method {args.color_method} not found!")
            exit(1)

        processed_image = True
        print(f"Color adjusted image in {time()-color_tic} seconds")

    if args.super_resolution:
        from super_resolution.super_resolution import super_resolution

        print("Super resolution")
        super_tic = time()
        image = super_resolution(
            image, args.super_resolution_method, args.super_resolution_scale
        )

        processed_image = True
        print(f"Super resolution image in {time()-super_tic} seconds")

    if args.shrink_images:
        from super_resolution.super_resolution import super_resolution

        print("Resize shrunk image")
        super_tic = time()
        image = super_resolution(
            image, args.super_resolution_method, 2
        )

        processed_image = True
        print(f"Super resolution image in {time()-super_tic} seconds")

    if processed_image:
        output_image = os.path.join(
            args.input_dir, f"main_processed.{args.interal_image_extension}"
        )
        print(f"Saved {output_image}")
        cv2.imwrite(output_image, image)
