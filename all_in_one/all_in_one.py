import argparse, os, cv2, rawpy, imageio
import numpy as np
from time import time
from skimage.exposure import is_low_contrast
from requests import get
from collections import OrderedDict
from pathlib import Path

def process_raw(dng_file):
    with rawpy.imread(dng_file) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    

# Create a numpy array for all the dng images in the folder
def createNumpyArray(path):
    file_list = os.listdir(path)
    dng_file_list = [os.path.join(path, x)
                        for x in file_list if x.endswith('.dng')]

    # Read image to get dimensions
    second_image = process_raw(dng_file_list[1])
    h, w, _ = second_image.shape
    
    # Create numpy array
    numpy_array = np.zeros((len(dng_file_list), h, w, 3), dtype=np.uint8)
    
    # Read all images into numpy array
    for i, file in enumerate(dng_file_list):
        if i==1:
            numpy_array[i] = second_image
        else:
            numpy_array[i] = process_raw(file)

    return numpy_array
    
# Filter out images with low contrast from numpy array
def filterLowContrast(numpy_array):
    filtered_array = []
    for i, image in enumerate(numpy_array):
        if is_low_contrast(image, fraction_threshold=0.05, lower_percentile=10, upper_percentile=90, method='linear'):
            print(f"Image {i} is low contrast")
        else:
            filtered_array.append(image)
    return filtered_array



# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECC(numpy_array):
    M = np.eye(3, 3, dtype=np.float32)
    
    first_image = None
    stacked_image = None

    for _, image in enumerate(numpy_array):
        #image = cv2.imread(file, 1).astype(np.float32) / 255
        imageF = image.astype(np.float32) / 255
        if first_image is None:
            # convert to gray scale floating point image
            #first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #stacked_image = image
            first_image = cv2.cvtColor(imageF, cv2.COLOR_BGR2GRAY)
            stacked_image = imageF
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(
                imageF, cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
            w, h, _ = imageF.shape
            # Align image to first image
            image_align = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += image_align

    stacked_image /= len(numpy_array)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(numpy_array):

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for _, image in enumerate(numpy_array):
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
            # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF

    stacked_image /= len(numpy_array)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image

# fast denoise image
def fastDenoiseImage(image, denoise_amount):
    image = cv2.fastNlMeansDenoisingColored(image, None, denoise_amount, denoise_amount, 7, 21)
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

# fddnet denoise image
def fddnetDenoiseImage(image, denoise_amount):
    import torch

    from fddnet_utils import utils_image as fddnet_util
    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = denoise_amount            # noise level for noisy image
    noise_level_model = noise_level_img         # noise level for model
    model_name = 'ffdnet_color'                 # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    need_degradation = True                     # default: True
    url = f"https://github.com/cszn/KAIR/releases/download/v1.0/{model_name}.pth"

    task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    sf = 1                    # unused for denoising
    if 'color' in model_name:
        n_channels = 3        # setting for color image
        nc = 96               # setting for color image
        nb = 12               # setting for color image
    else:
        n_channels = 1        # setting for grayscale image
        nc = 64               # setting for grayscale image
        nb = 15               # setting for grayscale image
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
    
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_dir = Path( __file__ ).parent.absolute()
    model_path = os.path.join( model_dir, model_name+'.pth')

    if not os.path.exists(model_path):
        print("Downloading model...")
        downloader(url, model_path, model_dir)
        print("Download Complete")

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from fddnet_models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)


    # ------------------------------------
    # (1) img_L
    # ------------------------------------

    img_L = fddnet_util.uint2single(image)

    if need_degradation:  # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
        if use_clip:
            img_L = fddnet_util.uint2single(fddnet_util.single2uint(img_L))

    img_L = fddnet_util.single2tensor4(img_L)
    img_L = img_L.to(device)

    sigma = torch.full((1,1,1,1), noise_level_model/255.).type_as(img_L)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------

    img_E = model(img_L, sigma)
    img_E = fddnet_util.tensor2uint(img_E)

    # ------------------------------------
    # save results
    # ------------------------------------

    return img_E

# super resolution image
def super_resolution(image, method, scale):
    model_path = Path( __file__ ).parent.absolute()
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

# ===== MAIN =====
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process Image')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument(
        '--stack_method', help='Stacking method ORB (faster) or ECC (more precise)', choices=['ORB', 'ECC'], default='ECC')
    parser.add_argument('--filter_contrast',
                        help='Filter low contrast images', action='store_true')
    parser.add_argument('--show', help='Show result image',
                        action='store_true')
    parser.add_argument('--denoise_method', help='Denoise image', choices=['fast', 'fddnet', 'ircnn', 'none'], default='none')
    parser.add_argument('--denoise_amount', help='Denoise amount', type=int, default=2)
    parser.add_argument('--super_resolution', help='Super resolution', action='store_true')
    parser.add_argument(
        "--super_resolution_method",
        help="Super Resolution method",
        choices=["ESPCN", "FSRCNN"],
        default="ESPCN",
    )
    parser.add_argument(
        "--super_resolution_scale", help="Super Resolution Scale", choices=[2, 4], type=int, default=2
    )
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print(f"ERROR {image_folder} not found!")
        exit()
    
    if image_folder.endswith('/'):
        image_folder = image_folder[:-1]

    numpy_images = createNumpyArray(image_folder)
    
    if args.filter_contrast:
        print("Filtering low contrast images")
        numpy_images = filterLowContrast(numpy_images)
    
    print(f"Loaded {len(numpy_images)} images")

    tic = time()

    if args.stack_method == 'ECC':
        # Stack images using ECC method
        description = "Stacking images using ECC method"
        print(description)
        stacked_image = stackImagesECC(numpy_images)

    elif args.stack_method == 'ORB':
        # Stack images using ORB keypoint method
        description = "Stacking images using ORB method"
        print(description)
        stacked_image = stackImagesKeypointMatching(numpy_images)

    else:
        print(f"ERROR: method {args.stack_method} not found!")
        exit()

    if args.denoise_method != 'none':
        tic = time()
        print("Denoise image")

        if args.denoise_method == 'fast':
            stacked_image = fastDenoiseImage(stacked_image, args.denoise_amount)

        if args.denoise_method == "fddnet":
            stacked_image = fddnetDenoiseImage(stacked_image, args.denoise_amount)

        print(f"Denoised image in {time()-tic} seconds")

    if args.super_resolution:
        print("Super resolution")
        tic = time()
        stacked_image = super_resolution(stacked_image, args.super_resolution_method, args.super_resolution_scale)
        print(f"Super resolution image in {time()-tic} seconds")

    print(f"Saved {args.output_image}")
    cv2.imwrite(str(args.output_image), stacked_image)