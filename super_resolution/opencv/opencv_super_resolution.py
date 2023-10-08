import cv2, os
from pathlib import Path
from utils.utils import downloader


# super resolution image
def opencv_super_resolution(image, method, scale):
    model_path = Path(__file__).parent.absolute()
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    if method == "ESPCN":
        model_file = f"{model_path}/ESPCN_x{scale}.pb"
        url = f"https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x{scale}.pb"
        model = "espcn"
    elif method == "FSRCNN":
        model_file = f"{model_path}/FSRCNN_x{scale}.pb"
        url = f"https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x{scale}.pb"
        model = "fsrcnn"
    else:
        raise Exception("Super Resolution: Method not supported")

    # If path does not exist, download the model
    if not os.path.exists(model_file):
        print("Downloading model...")
        downloader(url, model_file, model_path)
        print("Download Complete")

    sr.readModel(model_file)
    sr.setModel(model, scale)
    print("Running Super Sampling")
    result = sr.upsample(image)

    return result
