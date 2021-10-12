#!/bin/bash

declare -a scales=(2 4)

for scale in "${scales[@]}"; do
    wget -O "super_resolution/opencv_super_resolution/ESPCN_x${scale}.pb" "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x${scale}.pb"
    wget -O "super_resolution/opencv_super_resolution/FSRCNN_x${scale}.pb" "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x${scale}.pb"
done

declare -a ffdnet_models=( "ffdnet_gray" "ffdnet_color" "ffdnet_color_clip" "ffdnet_gray_clip" )
declare -a ircnn_models=( "ircnn_gray" "ircnn_color" )

for model in "${ffdnet_models[@]}"; do
    wget -O "denoise/ffdnet/${model}.pth" "https://github.com/cszn/KAIR/releases/download/v1.0/${model}.pth"
done

for model in "${ircnn_models[@]}"; do
    wget -O "denoise/ircnn/${model}.pth" "https://github.com/cszn/KAIR/releases/download/v1.0/${model}.pth"
done
