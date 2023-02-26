import sys
import argparse

import numpy as np

import torch
import torchvision.transforms.functional as F

from torchvision.transforms import (
    Normalize,
    Compose,
    InterpolationMode,
    ToTensor,
    Resize,
    CenterCrop,
    ToPILImage,
)

import torch
import bentoml
import time

import whisper


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-t", "--test", action="store_true", help="verbose output")

    # ViT - B - 16
    # openai
    parser.add_argument("-m", "--model", default="ViT-B-16")
    parser.add_argument("-p", "--pretrained", default="openai")
    # parser.add_argument("-m", "--model", default="xlm-roberta-large-ViT-H-14")
    # parser.add_argument("-p", "--pretrained", default="frozen_laion5b_s13b_b90k")
    parser.add_argument("-f", "--format", default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    return args




class WhisperWrapper(torch.nn.Module):
    def __init__(self, whisper, format=torch.float32) -> None:
        super().__init__()
        self.whisper = whisper


    def forward(self, audio_samples):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        audio = whisper.pad_or_trim(audio_samples.flatten()).to(device)
        mels = whisper.log_mel_spectrogram(audio)
        
        options = {}
        if not  torch.cuda.is_available():
            options["fp16"] = False
        options = whisper.DecodingOptions(**options)
        results = self.whisper.decode(mels, options)
        return results
        


def main():
    args = parse_args()

    # if args.format == "float16":
    #     format = torch.float16
    # elif args.format == "float32":
    #     format = torch.float32
    # elif args.format == "bfloat16":
    #     format = torch.bfloat16

    # print(open_clip.list_pretrained())

    model = whisper.load_model("base")
    wrapper = WhisperWrapper(model).eval()


    bentoml.pytorch.save_model(
        f"whisper",
        wrapper,
        signatures={"__call__": {"batchable": False}},
        labels={"model": f"whisper"},
    )


    ## Test
    # if args.test:
    #     np.random.seed(42)
    #     start = time.time()

    #     num_rounds = 20
    #     for x in range(num_rounds):

    #         test_image = (np.random.rand(1, 512, 768, 3) * 255).astype(np.int16)
    #         image = image_preprocessor(test_image)
    #         print(image.shape)

    #         image = image.to(args.device)

    #         test_text = "An image of a cat"
    #         text = tokenizer(test_text)
    #         text = text.to(args.device)
    #         print(text.shape)

    #         image_output = image_wrapper(image)
    #         print(image_output.shape)
    #         print(image_output[0, :10])

    #         print(text.device)

    #         text_output = text_wrapper(text)
    #         print(text_output.shape)
    #         print(text_output[0, :10])

    #         end = time.time()
    #         print((end - start) / (x + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
