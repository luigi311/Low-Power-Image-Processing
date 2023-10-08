#!/usr/bin/env bash

declare -a scales=(2 4)
declare -a ffdnet_models=( "ffdnet_gray" "ffdnet_color" "ffdnet_color_clip" "ffdnet_gray_clip" )
declare -a ircnn_models=( "ircnn_gray" "ircnn_color" )
declare -a image_adaptive_3dlut_models=( "LUTs" "LUTs_unpaired" "classifier" "classifier_unpaired" )

mkdir -p super_resolution/ denoise/ffdnet denoise/ircnn color/image_adaptive_3dlut/pretrained_models/sRGB

for scale in "${scales[@]}"; do
    wget -O "super_resolution/opencv/ESPCN_x${scale}.pb" "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x${scale}.pb"
    wget -O "super_resolution/opencv/FSRCNN_x${scale}.pb" "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x${scale}.pb"
done

for model in "${ffdnet_models[@]}"; do
    wget -O "denoise/ffdnet/${model}.pth" "https://github.com/cszn/KAIR/releases/download/v1.0/${model}.pth"
done

for model in "${ircnn_models[@]}"; do
    wget -O "denoise/ircnn/${model}.pth" "https://github.com/cszn/KAIR/releases/download/v1.0/${model}.pth"
done

for model in "${image_adaptive_3dlut_models[@]}"; do
    wget -O "color/image_adaptive_3dlut/pretrained_models/sRGB/${model}.pth" "https://github.com/HuiZeng/Image-Adaptive-3DLUT/raw/master/pretrained_models/sRGB/${model}.pth"
done