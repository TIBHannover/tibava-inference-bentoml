import onnx
import sys
import argparse

import numpy as np


import torch
import bentoml


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-t", "--test", action="store_true", help="verbose output")
    parser.add_argument("-p", "--path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = torch.jit.load(args.path)

    bentoml.torchscript.save_model(
        f"insightface_detector",
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"insightface_detector"},
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
