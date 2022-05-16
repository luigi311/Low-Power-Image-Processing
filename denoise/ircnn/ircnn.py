import os.path
import logging
from requests import get  # to make GET request

import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat

import torch

from utils import utils_model
from utils import utils_image as util


"""
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/IRCNN

@inproceedings{zhang2017learning,
title={Learning deep CNN denoiser prior for image restoration},
author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
booktitle={IEEE conference on computer vision and pattern recognition},
pages={3929--3938},
year={2017}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
by Luigi311 (12/Oct./2021)
"""


def downloader(url, file_name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input", help="Input images")
    parser.add_argument("--noise", help="noise amount", type=int, default=15)
    parser.add_argument(
        "--model",
        help="model",
        type=str,
        choices=["ircnn_gray", "ircnn_color"],
        default="ircnn_color",
    )
    parser.add_argument(
        "--model_path",
        help="Path to model",
        type=str,
        default=os.path.dirname(__file__),
    )
    parser.add_argument("--show", help="Show result image", action="store_true")
    args = parser.parse_args()

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    noise_level_img = args.noise  # noise level for noisy image
    model_name = args.model  # 'ircnn_gray' | 'ircnn_color'
    need_degradation = True  # default: True
    x8 = False  # default: False, x8 to boost performance
    show_img = args.show  # default: False
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

    border = (
        sf if task_current == "sr" else 0
    )  # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(args.model_path, model_name + ".pth")

    if not os.path.exists(model_path):
        print("Downloading model...")
        downloader(url, model_path, args.model_path)
        print("Download Complete")

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = args.input  # L_path, for Low-quality images
    H_path = L_path  # H_path, for High-quality images
    E_path = args.input  # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True

    need_H = True if H_path is not None else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------
    # load model
    # ----------------------------------------
    model25 = torch.load(model_path)
    from models.network_dncnn import IRCNN as net

    model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
    model.load_state_dict(model25[str(current_idx)], strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))

    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        if need_degradation:  # degradation process
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img / 255.0, img_L.shape)

        util.imshow(
            util.single2uint(img_L),
            title="Noisy image with noise level {}".format(noise_level_img),
        ) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        if not x8:
            img_E = model(img_L)
        else:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_E, os.path.join(E_path, img_name + "_denoised" + ext))


if __name__ == "__main__":

    main()
