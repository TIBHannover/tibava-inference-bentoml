import sys
import argparse

import numpy as np


import torch
import bentoml
import time

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-t", "--test", action="store_true", help="verbose output")
    parser.add_argument("-p", "--path")    
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = torch.jit.load(args.path, map_location=torch.device(args.device))

    bentoml.torchscript.save_model(
        f"transnet",
        model,
        signatures={"__call__": {"batchable": False, "batch_dim": 0}},
        labels={"model": f"transnet"},
    )
    if args.test:
        np.random.seed(42)
        test_image = (np.random.rand(1, 100, 27, 48, 3) * 255).astype(np.uint8)
        print(model(torch.as_tensor(test_image)))

        runner = bentoml.torchscript.get("transnet:latest").to_runner()
        runner.init_local()
        print(runner.run(torch.as_tensor(test_image)))  # .to("cuda:0")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
