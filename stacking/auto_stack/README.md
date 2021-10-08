# Automatic Image Stacking in OpenCV

Source: <https://github.com/maitek/image_stacking>

## Changes

-   Add filter_contrast to avoid using corrupt images (useful for pinephone)

## Description

Image stacking is a technique in digital photograpy where multiple photos from the same scene are "stacked" together
in order to reduce image noise caused by a high ISO values. This allows for creating a low noise image at low light conditions. In order to fuse multiple images together the pixels in all photos needs to aligned with each other,
which is usually not possible when taking images without a tripod. This python script shows an example of stack multiple images together by first automatically alligning and then calculating a the fused images as a pixel-wise mean of the alligned images.

To see the power of image stacking, check out the before and after images below. Left is a single image taken at in low light, and right is 8 images stacked from the same scene stacked toghether. The more images you have the more the noise can be reduced.

In order to allign the images the perspective transforms between the first images and every other images are to be estimated. The perspective transform can be estimated either using [ORB Keypoint matching](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html) (faster) or
[ECC maximization](http://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#findtransformecc) (more precise).

## Requirements

-   Python 2 or 3
-   OpenCV 3.0 or higher

## Example Usage

### ORB method (faster)

```bash
  python auto_stack.py images/ result.jpg --method ORB
```

### ECC method (more precise)

```bash
  python auto_stack.py images/ result.jpg --method ECC
```
