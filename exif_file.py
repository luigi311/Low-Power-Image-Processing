import argparse, traceback, json
import exif, exifread

from time import time


def setup_args():
    parser = argparse.ArgumentParser(
        description="Copy exif data from one image to another"
    )
    parser.add_argument("--image_source", help="The image to copy exif data from")
    parser.add_argument("--image_destination", help="The image to place exif data into")
    parser.add_argument("--json", help="The json file to copy exif data from")
    parser.add_argument("--output", help="The json file to copy exif data to")

    return parser.parse_args()


def generate_exif(image: str) -> dict:
    with open(image, "rb") as image_file:
        tags = exifread.process_file(image_file, details=False)

    # Convert all tags to strings
    for key, value in tags.items():
        tags[key] = str(value)

    return tags


def read_json(json_file: str) -> dict:
    with open(json_file, "r") as json_file:
        return json.load(json_file)


def write_json(json_file: str, data: dict):
    with open(json_file, "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_to_image(image: str, tags: dict):
    print(f"Writing exif data to {image}")

    with open(image, "rb") as image_file:
        file_tags = exif.Image(image_file)

    if tags.get("Image Make"):
        file_tags.make = tags.get("Image Make")
    if tags.get("Image Model"):
        file_tags.model = tags.get("Image Model")
    if tags.get("Image Software"):
        file_tags.software = tags.get("Image Software")

    if tags.get("Image DateTime"):
        file_tags.datetime = tags.get("Image DateTime")
    if tags.get("EXIF DateTimeOriginal"):
        file_tags.datetime_original = tags.get("EXIF DateTimeOriginal")
    if tags.get("EXIF DateTimeDigitized"):
        file_tags.datetime_digitized = tags.get("EXIF DateTimeDigitized")

    if tags.get("EXIF ExposureTime"):
        file_tags.exposure_time = tags.get("EXIF ExposureTime")
    if tags.get("EXIF FNumber"):
        file_tags.f_number = tags.get("EXIF FNumber")
    if tags.get("EXIF ISOSpeedRatings"):
        file_tags.photographic_sensitivity = tags.get("EXIF ISOSpeedRatings")
    if tags.get("EXIF FocalLength"):
        file_tags.focal_length = tags.get("EXIF FocalLength")
    if tags.get("EXIF FocalLengthIn35mmFilm"):
        file_tags.focal_length_in_35mm_film = tags.get("EXIF FocalLengthIn35mmFilm")
    if tags.get("EXIF ColorSpace"):
        file_tags.color_space = tags.get("EXIF ColorSpace")

    if "not" in tags.get("EXIF Flash"):
        file_tags.flash = False
    else:
        file_tags.flash = True

    with open(image, "wb") as image_file:
        image_file.write(file_tags.get_file())

    print(f"Successfully wrote exif data to {image}")


def main(args):
    print("Starting exif copy")

    if args.image_source:
        tags = generate_exif(args.image_source)
    elif args.json:
        tags = read_json(args.json)
    else:
        raise Exception("No image or json file specified")

    if args.output:
        write_json(args.output, tags)
    elif args.image_destination:
        write_to_image(args.image_destination, tags)
    else:
        raise Exception("No output or image_destination specified")

if __name__ == "__main__":
    try:
        total_tic = time()

        args = setup_args()
        main(args)

        print(f"Total {time() - total_tic} seconds")

    except Exception as error:
        if isinstance(error, list):
            for message in error:
                print(message)
        else:
            print(error)

        print(traceback.format_exc())
