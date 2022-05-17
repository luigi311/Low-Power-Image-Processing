import argparse
import torch
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from color.image_adaptive_3dlut.models import *
import color.image_adaptive_3dlut.torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF


def image_adaptive_3dlut(image, input_color_space):

    model_dir = (
        f"{Path(__file__).parent.absolute()}/pretrained_models/{input_color_space}"
    )

    # use gpu when detect cuda
    # cuda = True if torch.cuda.is_available() else False
    cuda = False
    # Tensor type
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.FloatTensor

    criterion_pixelwise = torch.nn.MSELoss()
    LUT0 = Generator3DLUT_identity()
    LUT1 = Generator3DLUT_zero()
    LUT2 = Generator3DLUT_zero()
    # LUT3 = Generator3DLUT_zero()
    # LUT4 = Generator3DLUT_zero()
    classifier = Classifier()
    trilinear_ = TrilinearInterpolation()

    if cuda:
        LUT0 = LUT0.cuda()
        LUT1 = LUT1.cuda()
        LUT2 = LUT2.cuda()
        # LUT3 = LUT3.cuda()
        # LUT4 = LUT4.cuda()
        classifier = classifier.cuda()
        criterion_pixelwise.cuda()

    # Load pretrained models
    LUTs = torch.load(f"{model_dir}/LUTs.pth", map_location=torch.device("cpu"))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    # LUT3.load_state_dict(LUTs["3"])
    # LUT4.load_state_dict(LUTs["4"])
    LUT0.eval()
    LUT1.eval()
    LUT2.eval()
    # LUT3.eval()
    # LUT4.eval()
    classifier.load_state_dict(
        torch.load(f"{model_dir}/classifier.pth", map_location=torch.device("cpu"))
    )
    classifier.eval()

    def generate_LUT(img):

        pred = classifier(img).squeeze()

        LUT = (
            pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT
        )  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

        return LUT

    # ----------
    #  test
    # ----------
    # read image and transform to tensor
    if input_color_space == "sRGB":
        img = TF.to_tensor(image).type(Tensor)
    img = img.unsqueeze(0)

    LUT = generate_LUT(img)

    # generate image
    # scale im between -1 and 1 since its used as grid input in grid_sample
    img = (img - 0.5) * 2.0

    # grid_sample expects NxDxHxWx3 (1x1xHxWx3)
    img = img.permute(0, 2, 3, 1)[:, None]

    # add batch dim to LUT
    LUT = LUT[None]

    # grid sample
    result = F.grid_sample(
        LUT, img, mode="bilinear", padding_mode="border", align_corners=True
    )

    # drop added dimensions and permute back
    result = result[:, :, 0].permute(0, 2, 3, 1)

    # save image
    ndarr = (
        result.squeeze()
        .mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    # fix order from (1944, 3, 2592) to (2592, 1944, 3)
    ndarr = ndarr.transpose(2, 0, 1)

    return ndarr
