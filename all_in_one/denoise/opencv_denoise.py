import cv2

# fast denoise image
def fastDenoiseImage(image, denoise_amount):
    image = cv2.fastNlMeansDenoisingColored(
        image, None, denoise_amount, denoise_amount, 7, 21
    )
    return image
