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

    model = onnx.load(args.path)

    bentoml.onnx.save_model(
        f"deepface_emotion",
        model,
        signatures={"run": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"deepface_emotion"},
    )

    if args.test:
        runner = bentoml.onnx.get("deepface_emotion:latest").to_runner()
        runner.init_local()
        # import onnxruntime as ort

        # ort_sess = ort.InferenceSession(args.path, providers=["CPUExecutionProvider"])
        np.random.seed(42)
        test_image = (np.random.rand(1, 48, 48, 1) * 255).astype(np.uint8)
        output = runner.run.run(test_image)
        if isinstance(output, (list, set, tuple)):
            for x in output:
                print(x.shape)
        else:

            print(output.shape)

    return 0


if __name__ == "__main__":
    sys.exit(main())
