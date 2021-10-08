# Low Power Image Processing

This repo is a collection of different image processing algorithms that are
able to be used on low powered devices in a reasonable amount of time. These can be used a image processing pipeline to improve the quality of the end result.

Recommended to process an image the normal route and then pick and chose what to use from here based on your device to create a seperate "improved" image

## Devices tested on

-   Pinephone Convergence Edition (Pine64)

## Algorithms

-   stacking
    -   auto_stacking: Stack images in a single image to reduce noise and import quality

-   super_resolution
    -   opencv_super_resolution: Super resolution using OpenCV built-in methods
