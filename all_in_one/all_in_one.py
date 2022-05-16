import argparse, os, cv2
from time import time

from utils.utils import createNumpyArray, filterLowContrast
from stacking.stacking import stackImagesECC, stackImagesKeypointMatching


# ===== MAIN =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Image")
    parser.add_argument("input_dir", help="Input directory of images ()")
    parser.add_argument("output_image", help="Output image name")
    parser.add_argument(
        "--stack_method",
        help="Stacking method ORB (faster) or ECC (more precise)",
        choices=["ORB", "ECC"],
        default="ECC",
    )
    parser.add_argument(
        "--filter_contrast", help="Filter low contrast images", action="store_true"
    )
    parser.add_argument("--show", help="Show result image", action="store_true")
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
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print(f"ERROR {image_folder} not found!")
        exit()

    if image_folder.endswith("/"):
        image_folder = image_folder[:-1]

    loading_tic = time()
    numpy_images = createNumpyArray(image_folder)

    if args.filter_contrast:
        print("Filtering low contrast images")
        numpy_images = filterLowContrast(numpy_images)

    print(f"Loaded {len(numpy_images)} images in {time()-loading_tic} seconds")

    stack_tic = time()

    if args.stack_method == "ECC":
        # Stack images using ECC method
        description = "Stacking images using ECC method"
        print(description)
        stacked_image = stackImagesECC(numpy_images)

    elif args.stack_method == "ORB":
        # Stack images using ORB keypoint method
        description = "Stacking images using ORB method"
        print(description)
        stacked_image = stackImagesKeypointMatching(numpy_images)

    else:
        print(f"ERROR: method {args.stack_method} not found!")
        exit(1)

    print(f"Stacked images in {time() - stack_tic} seconds")

    if args.denoise_method != "none":
        denoise_tic = time()
        print("Denoise image")

        if args.denoise_method == "fast":
            from denoise.opencv_denoise import fastDenoiseImage

            stacked_image = fastDenoiseImage(stacked_image, args.denoise_amount)

        elif args.denoise_method == "fddnet":
            from denoise.fddnet import fddnetDenoiseImage

            stacked_image = fddnetDenoiseImage(stacked_image, args.denoise_amount)

        elif args.denoise_method == "ircnn":
            from denoise.ircnn import ircnnDenoiseImage

            stacked_image = ircnnDenoiseImage(stacked_image, args.denoise_amount)

        else:
            print(f"ERROR: method {args.denoise_method} not found!")
            exit(1)

        print(f"Denoised image in {time()-denoise_tic} seconds")

    if args.super_resolution:
        from super_resolution.super_resolution import super_resolution

        print("Super resolution")
        super_tic = time()
        stacked_image = super_resolution(
            stacked_image, args.super_resolution_method, args.super_resolution_scale
        )
        print(f"Super resolution image in {time()-super_tic} seconds")

    print(f"Saved {args.output_image}")
    cv2.imwrite(str(args.output_image), stacked_image)
