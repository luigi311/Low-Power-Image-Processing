def denoiser(image, method, amount):
    try:
        print(f"Denoising image via {method} with amount {amount}")

        if method == "fast":
            from denoise.opencv_denoise.opencv_denoise import fastDenoiseImage

            image = fastDenoiseImage(image, amount)

        elif method == "fddnet":
            from denoise.fddnet.fddnet import fddnetDenoiseImage

            image = fddnetDenoiseImage(image, amount)

        elif method == "ircnn":
            from denoise.ircnn.ircnn import ircnnDenoiseImage

            image = ircnnDenoiseImage(image, amount)

        else:
            raise Exception(f"ERROR: Denoise method {method} not found!")

        return image

    except Exception as e:
        raise Exception(e)
