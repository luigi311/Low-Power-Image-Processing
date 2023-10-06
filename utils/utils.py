import os

from requests import get
from concurrent.futures import ThreadPoolExecutor


def files(path):
    # Check if the path exists and is a directory
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError(f"ERROR {path} is not a valid directory.")

    # Remove trailing slash from path if present
    if path.endswith("/"):
        path = path[:-1]

    # Get a list of all files in the directory
    file_list = os.listdir(path)

    # Filter the list to only include dng, tiff, and hdf5 files
    process_file_list = [
        os.path.join(path, x) for x in file_list if x.endswith(("dng", "tiff"))
    ]

    # Check if there are any dng or tiff files in the directory
    dng_files = [x for x in process_file_list if x.endswith("dng")]
    tiff_files = [x for x in process_file_list if x.endswith("tiff")]

    return dng_files, tiff_files


def downloader(url, file_name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


def future_thread_executor(args: list, workers: int = -1):
    futures_list = []
    results = []

    if workers == -1:
        workers = os.cpu_count() - 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for arg in args:
            # * arg unpacks the list into actual arguments
            futures_list.append(executor.submit(*arg))

        for future in futures_list:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise Exception(e)

    return results
