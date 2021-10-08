# Super Resolution using buildin opencv functions

Source: <https://learnopencv.com/super-resolution-in-opencv/>

## Changes

-   Implemented for easier usage on single images
-   Only implement those that run well on low performace devices

## Requirements

-   Python
-   OpenCV

## Usage

### FSRCNN 2x

```bash
python opencv_super_resolution.py  input_image.png output_image.png --method FSRCNN --scale 2
```

### ESPCN 4x

```bash
python opencv_super_resolution.py  input_image.png output_image.png --method ESPCN --scale 4
```
