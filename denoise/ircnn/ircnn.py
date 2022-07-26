import os, torch
import numpy as np
from pathlib import Path

from utils.utils import downloader
from denoise.ircnn.ircnn_models.network_dncnn import IRCNN as net
from denoise.ircnn.ircnn_utils import utils_image as ircnn_util
from denoise.ircnn.ircnn_utils import utils_model as ircnn_model


def ircnnDenoiseImage(image, denoise_amount):
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    noise_level_img = denoise_amount  # noise level for noisy image
    model_name = "ircnn_color"  # 'ircnn_gray' | 'ircnn_color'
    need_degradation = True  # default: True
    x8 = False  # default: False, x8 to boost performance
    current_idx = min(
        24, int(np.ceil(noise_level_img / 2) - 1)
    )  # current_idx+1 th denoiser
    url = f"https://github.com/cszn/KAIR/releases/download/v1.0/{model_name}.pth"

    task_current = "dn"  # fixed, 'dn' for denoising | 'sr' for super-resolution
    sf = 1  # unused for denoising
    if "color" in model_name:
        n_channels = 3  # fixed, 1 for grayscale image, 3 for color image
    else:
        n_channels = 1  # fixed for grayscale image

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
    model25 = torch.load(model_path)
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
    model.load_state_dict(model25[str(current_idx)], strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # ------------------------------------
    # (1) img_L
    # ------------------------------------
    img_L = ircnn_util.uint2single(image)

    if need_degradation:  # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img / 255.0, img_L.shape)

    img_L = ircnn_util.single2tensor4(img_L)
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------
    if not x8:
        img_E = model(img_L)
    else:
        img_E = ircnn_model.test_mode(model, img_L, mode=3)

    img_E = ircnn_util.tensor2uint(img_E)

    # ------------------------------------
    # save results
    # ------------------------------------

    return img_E
