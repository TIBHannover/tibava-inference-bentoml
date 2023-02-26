import os
import sys
import re
import argparse
import imageio
import requests
import json


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-i", "--image_path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image = imageio.imread(args.image_path)

    print(image.shape)
    data = json.dumps(image.tolist())

    image_preprocessed = requests.post(
        "http://127.0.0.1:3000/clip_image_preprocessor",
        headers={"content-type": "application/json"},
        data=data,
    ).text

    print(json.loads(image_preprocessed))
    image_encoded = requests.post(
        "http://127.0.0.1:3000/clip_image_encoder",
        headers={"content-type": "application/json"},
        data=image_preprocessed,
    ).text

    print(json.loads(image_encoded))

    return 0


if __name__ == "__main__":
    sys.exit(main())
