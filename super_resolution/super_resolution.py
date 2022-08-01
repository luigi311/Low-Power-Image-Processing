from super_resolution.opencv.opencv_super_resolution import opencv_super_resolution


def super_resolution(image, method, scale):
    if method == "ESPCN":
        return opencv_super_resolution(image, method, scale)
    elif method == "FSRCNN":
        return opencv_super_resolution(image, method, scale)
    else:
        print("Method not supported")
        return
