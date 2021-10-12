#!/bin/bash

declare -a scales=(2 4)

for scale in "${scales[@]}"; do
    wget -O "super_resolution/opencv_super_resolution/ESPCN_x${scale}.pb" "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x${scale}.pb"
    wget -O "super_resolution/opencv_super_resolution/FSRCNN_x${scale}.pb" "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x${scale}.pb"
done
