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
    parser.add_argument("-f", "--format", default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = torch.jit.load(args.path, map_location=torch.device(args.device))

    bentoml.torchscript.save_model(
        f"insightface_detector",
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"insightface_detector"},
    )

    if args.test:

        model.to(args.device)
        np.random.seed(42)
        test_image = (np.random.rand(1, 640, 640, 3) * 255).astype(np.uint8)
        output = model(torch.as_tensor(test_image).to(args.device))
        if isinstance(output, (list, set, tuple)):
            for x in output:
                print(x.shape)
        else:

            print(output.shape)

        # runner = bentoml.torchscript.get("transnet:latest").to_runner()
        # runner.init_local()
        # print(runner.run(torch.as_tensor(test_image)))  # .to("cuda:0")))

    return 0


if __name__ == "__main__":
    sys.exit(main())
