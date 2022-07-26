import torch, os
import numpy as np
from pathlib import Path

from denoise.fddnet.fddnet_utils import utils_image as fddnet_util
from denoise.fddnet.fddnet_models.network_ffdnet import FFDNet as net
from utils.utils import downloader


# fddnet denoise image
def fddnetDenoiseImage(image, denoise_amount):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = denoise_amount  # noise level for noisy image
    noise_level_model = noise_level_img  # noise level for model
    model_name = "ffdnet_color"  # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    need_degradation = True  # default: True
    url = f"https://github.com/cszn/KAIR/releases/download/v1.0/{model_name}.pth"

    if "color" in model_name:
        n_channels = 3  # setting for color image
        nc = 96  # setting for color image
        nb = 12  # setting for color image
    else:
        n_channels = 1  # setting for grayscale image
        nc = 64  # setting for grayscale image
        nb = 15  # setting for grayscale image
    if "clip" in model_name:
        use_clip = True  # clip the intensities into range of [0, 1]
    else:
        use_clip = False

    model_dir = Path(__file__).parent.absolute()
    model_path = os.path.join(model_dir, model_name + ".pth")

    if not os.path.exists(model_path):
        print("Downloading model...")
        downloader(url, model_path, model_dir)
        print("Download Complete")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode="R")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # ------------------------------------
    # (1) img_L
    # ------------------------------------

    img_L = fddnet_util.uint2single(image)

    if need_degradation:  # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img / 255.0, img_L.shape)
        if use_clip:
            img_L = fddnet_util.uint2single(fddnet_util.single2uint(img_L))

    img_L = fddnet_util.single2tensor4(img_L)
    img_L = img_L.to(device)

    sigma = torch.full((1, 1, 1, 1), noise_level_model / 255.0).type_as(img_L)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------

    img_E = model(img_L, sigma)
    img_E = fddnet_util.tensor2uint(img_E)

    # ------------------------------------
    # save results
    # ------------------------------------

    return img_E
