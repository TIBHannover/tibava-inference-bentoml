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
        f"places_classification",
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"places_classification"},
    )

    if args.test:
        np.random.seed(42)
        test_image = (np.random.rand(1, 224, 224, 3) * 255).astype(np.uint8)
        output = model(torch.as_tensor(test_image))
        if isinstance(output, (list, set, tuple)):
            for x in output:
                print(x.shape)
        else:

            print(output.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
