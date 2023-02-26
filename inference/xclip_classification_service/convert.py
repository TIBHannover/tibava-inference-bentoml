import onnx
import sys
import argparse

import numpy as np


import torch
import bentoml
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-t", "--test", action="store_true", help="verbose output")
    parser.add_argument("--f16", action="store_true", help="verbose output")
    parser.add_argument("--sim_path")
    parser.add_argument("--text_path")
    parser.add_argument("--video_path")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(torch.cuda.is_available())
    sim_model = onnx.load(args.sim_path)

    if args.f16:
        sim_model = convert_float_to_float16(sim_model)

    bentoml.onnx.save_model(
        f"x_clip_sim",
        sim_model,
        signatures={"run": {"batchable": False}},
        labels={"model": f"x_clip_sim"},
    )

    text_model = onnx.load(args.text_path)

    if args.f16:
        text_model = convert_float_to_float16(text_model)

    bentoml.onnx.save_model(
        f"x_clip_text",
        text_model,
        signatures={"run": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"x_clip_text"},
    )

    video_model = onnx.load(args.video_path)

    if args.f16:
        video_model = convert_float_to_float16(video_model)

    bentoml.onnx.save_model(
        f"x_clip_video",
        video_model,
        signatures={"run": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"x_clip_video"},
    )

    if args.test:
        sim_runner = bentoml.onnx.get("x_clip_sim:latest").to_runner()
        print(dir(sim_runner))
        print(sim_runner)
        sim_runner.init_local()

        text_runner = bentoml.onnx.get("x_clip_text:latest").to_runner()
        text_runner.init_local()

        video_runner = bentoml.onnx.get("x_clip_video:latest").to_runner()
        video_runner.init_local()
        # import onnxruntime as ort

        # ort_sess = ort.InferenceSession(args.path, providers=["CPUExecutionProvider"])
        np.random.seed(42)
        test_clip = np.random.rand(30, 8, 3, 224, 224)
        text_batch_size = 10
        test_text = (np.random.randint(2000, size=text_batch_size * 77).reshape((text_batch_size, 77))).astype(np.uint8)

        video_output = video_runner.run.run(test_clip)
        if isinstance(video_output, (list, set, tuple)):
            for x in video_output:
                print(x.shape)
        else:

            print(video_output.shape)
        # {
        #                     "text_features": text_embedding,
        #                     "video_features": video_feature.embedding,
        #                     "image_features": image_feature.embedding,
        #                 },
        text_output = text_runner.run.run(test_text)
        if isinstance(text_output, (list, set, tuple)):
            for x in text_output:
                print(x.shape)
        else:

            print(text_output.shape)
        print(text_output.shape, video_output[0].shape, video_output[1].shape)
        sim_output = sim_runner.run.run(text_output, video_output[0], video_output[1])
        if isinstance(sim_output, (list, set, tuple)):
            for x in sim_output:
                print(x.shape)
        else:

            print(sim_output.shape)

    return 0


if __name__ == "__main__":
    sys.exit(main())
