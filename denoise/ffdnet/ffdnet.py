import os.path
import logging
import argparse
from requests import get  # to make GET request

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_image as util


'''
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/FFDNet

@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}

by Kai Zhang (12/Dec./2019)
by Luigi311 (12/Oct./2021)
'''

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
    parser.add_argument(
        "--noise", help="noise amount", type=int, default=10
    )
    parser.add_argument(
        "--model", help="model", type=str, choices=["ffdnet_gray", "ffdnet_color", "ffdnet_color_clip", "ffdnet_gray_clip"], default="ffdnet_color"
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

    noise_level_img = args.noise                # noise level for noisy image
    noise_level_model = noise_level_img         # noise level for model
    model_name = args.model                     # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
    need_degradation = True                     # default: True
    show_img = args.show                        # default: False
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
    model_path = os.path.join( args.model_path, model_name+'.pth')

    if not os.path.exists(model_path):
        print("Downloading model...")
        downloader(url, model_path, args.model_path)
        print("Download Complete")

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = args.input                           # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = args.input   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_ffdnet import FFDNet as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

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
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
            if use_clip:
                img_L = util.uint2single(util.single2uint(img_L))

        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        sigma = torch.full((1,1,1,1), noise_level_model/255.).type_as(img_L)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        img_E = model(img_L, sigma)
        img_E = util.tensor2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+"_denoised"+ext))

if __name__ == '__main__':

    main()
