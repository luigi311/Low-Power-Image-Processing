import argparse, os, cv2
import numpy as np
from requests import get  # to make GET request
from time import time


def downloader(url, file_name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


def super_resolution(image, method, scale, model_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    if method == "ESPCN":
        path = f"{model_path}/ESPCN_x{scale}.pb"
        url = f"https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x{scale}.pb"
        model = "espcn"
    elif method == "FSRCNN":
        path = f"{model_path}/FSRCNN_x{scale}.pb"
        url = f"https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x{scale}.pb"
        model = "fsrcnn"
    else:
        print("Method not supported")
        return

    # If path does not exist, download the model
    if not os.path.exists(path):
        print("Downloading model...")
        downloader(url, path, model_path)
        print("Download Complete")

    sr.readModel(path)
    sr.setModel(model, scale)
    print("Running Super Sampling")
    result = sr.upsample(image)

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_image", help="Input images")
    parser.add_argument("output_image", help="Output image name")
    parser.add_argument(
        "--method",
        help="Super Resolution method",
        choices=["ESPCN", "FSRCNN"],
        default="ESPCN",
    )
    parser.add_argument(
        "--scale", help="Super Resolution Scale", choices=[2, 4], type=int, default=2
    )
    parser.add_argument(
        "--model_path",
        help="Path to model",
        type=str,
        default=os.path.dirname(__file__),
    )
    parser.add_argument("--show", help="Show result image", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"ERROR {args.input_image} not found!")
        exit()

    image = cv2.imread(args.input_image)

    tic = time()
    print(f"Starting {args.method} {args.scale}x super sample")
    out_image = super_resolution(image, args.method, args.scale, args.model_path)
    cv2.imwrite(f"{args.output_image}", out_image)
    print(f"Super resolution image in {time()-tic} seconds")

    # Show image
    if args.show:
        description = f"{args.method} super resolution"
        cv2.imshow(description, out_image)
        cv2.waitKey(0)
