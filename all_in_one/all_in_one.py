import argparse, os, cv2
from time import time

from utils.utils import createNumpyArray, filterLowContrast

# Create main and do any processing if needed
def single_image(image, input_dir, image_extension="png"):
    print(f"Processing single image")

    output_image = os.path.join(input_dir, f"main.{image_extension}")
    print(f"Saved {output_image}")
    cv2.imwrite(output_image, image)

    return image


# ===== MAIN =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Image")
    parser.add_argument("input_dir", help="Input directory of images")
    parser.add_argument("--interal_image_extension", default="png", help="Extension of images to process")
    parser.add_argument("--single_image", help="Single image mode", action="store_true")
    parser.add_argument("--auto_stack", help="Auto stack images", action="store_true")
    parser.add_argument(
        "--stack_method",
        help="Stacking method ORB (faster) or ECC (more precise)",
        choices=["ORB", "ECC"],
        default="ECC",
    )
    parser.add_argument("--show", help="Show result image", action="store_true")
    parser.add_argument("--denoise_all", help="Denoise all images prior to stacking", action="store_true")
    parser.add_argument("--denoise_all_method", help="Denoise method for all images prior to stacking", choices=["fast", "fddnet", "ircnn"],
        default="fast",)
    parser.add_argument("--denoise_all_amount", help="Denoise amount for all images prior to stacking", type=int, default=2)
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

    processed_image = False

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print(f"ERROR {image_folder} not found!")
        exit()

    if image_folder.endswith("/"):
        image_folder = image_folder[:-1]

    loading_tic = time()
    numpy_images = createNumpyArray(image_folder)

    print("Filtering low contrast images")
    numpy_images = filterLowContrast(numpy_images)

    print(f"Loaded {len(numpy_images)} images in {time()-loading_tic} seconds")

    # Default to first image
    image = numpy_images[0]

    if args.single_image:
        # Create main image
        image = single_image(image, args.input_dir, args.interal_image_extension)
        if args.show:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    if args.denoise_all:
        denoise_all_tic = time()
        
        print("Denoise all image")
        numpy_images_denoised = []
        
        if args.denoise_all_method == "fast":
            from denoise.opencv_denoise import fastDenoiseImage
            
            for image in numpy_images:
                numpy_images_denoised.append(fastDenoiseImage(image, args.denoise_all_amount))

        elif args.denoise_all_method == "fddnet":
            from denoise.fddnet import fddnetDenoiseImage
            
            for image in numpy_images:
                numpy_images_denoised.append(fddnetDenoiseImage(image, args.denoise_all_amount))

        elif args.denoise_all_method == "ircnn":
            from denoise.ircnn import ircnnDenoiseImage

            for image in numpy_images:
                numpy_images_denoised.append(ircnnDenoiseImage(image, args.denoise_all_amount))

        else:
            print(f"ERROR: method {args.denoise_method} not found!")
            exit(1)

        numpy_images = numpy_images_denoised
        print(f"Denoised all images in {time()-denoise_all_tic} seconds")

    if args.auto_stack:
        stack_tic = time()

        if args.stack_method == "ECC":
            from stacking.stacking import stackImagesECC

            # Stack images using ECC method
            print("Stacking images using ECC method")
            image = stackImagesECC(numpy_images)

        elif args.stack_method == "ORB":
            from stacking.stacking import stackImagesKeypointMatching
            
            # Stack images using ORB keypoint method
            print("Stacking images using ORB method")
            image = stackImagesKeypointMatching(numpy_images)

        else:
            print(f"ERROR: method {args.stack_method} not found!")
            exit(1)
        
        processed_image = True
        print(f"Stacked images in {time() - stack_tic} seconds")

    if args.denoise_method != "none":
        denoise_tic = time()
        print("Denoise image")

        if args.denoise_method == "fast":
            from denoise.opencv_denoise import fastDenoiseImage

            image = fastDenoiseImage(image, args.denoise_amount)

        elif args.denoise_method == "fddnet":
            from denoise.fddnet import fddnetDenoiseImage

            image = fddnetDenoiseImage(image, args.denoise_amount)

        elif args.denoise_method == "ircnn":
            from denoise.ircnn import ircnnDenoiseImage

            image = ircnnDenoiseImage(image, args.denoise_amount)

        else:
            print(f"ERROR: method {args.denoise_method} not found!")
            exit(1)

        processed_image = True
        print(f"Denoised image in {time()-denoise_tic} seconds")

    if args.super_resolution:
        from super_resolution.super_resolution import super_resolution

        print("Super resolution")
        super_tic = time()
        image = super_resolution(
            image, args.super_resolution_method, args.super_resolution_scale
        )

        processed_image = True
        print(f"Super resolution image in {time()-super_tic} seconds")

    if processed_image:
        output_image = os.path.join(args.input_dir, f"main_processed.{args.interal_image_extension}")
        print(f"Saved {output_image}")
        cv2.imwrite(output_image, image)
