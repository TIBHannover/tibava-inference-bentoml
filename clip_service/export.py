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
import open_clip
import time


from open_clip import CLIP, CustomTextCLIP


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


class ClipCustomTextWrapper(torch.nn.Module):
    def __init__(self, clip, format=torch.float32) -> None:
        super().__init__()
        self.text = clip.text
        self.format = format
        self.to(format)

    def forward(self, text):
        with torch.no_grad():
            features = self.text(text)
            return torch.nn.functional.normalize(features, dim=-1)


class ClipTextWrapper(torch.nn.Module):
    def __init__(self, clip, format=torch.float32) -> None:
        super().__init__()
        self.transformer = clip.transformer
        self.vocab_size = clip.vocab_size
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        # self.attn_mask = clip.attn_mask
        self.format = format
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

        self.register_buffer("attn_mask", clip.attn_mask, persistent=False)
        self.to(format)

    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     self.attn_mask.to(*args, **kwargs)

    def forward(self, text):
        with torch.no_grad():
            cast_dtype = self.transformer.get_cast_dtype()
            print(f"DEVICE {self.attn_mask.device}")

            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.to(cast_dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return torch.nn.functional.normalize(x, dim=-1)


class ClipImageWrapper(torch.nn.Module):
    def __init__(self, clip, format=torch.float32) -> None:
        super().__init__()
        self.visual = clip.visual
        self.format = format
        self.to(format)

    def forward(self, image):
        with torch.no_grad():
            features = self.visual(image.to(self.format))
            return torch.nn.functional.normalize(features, dim=-1)


class TokenizerWrapper(torch.nn.Module):
    def __init__(self, tokenizer, format=torch.float32) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.format = format

    def forward(self, text):
        return self.tokenizer(text)


class ImagePreprozessorWrapper(torch.nn.Module):
    def __init__(self, clip, format=torch.float32) -> None:
        super().__init__()
        self.mean = None or getattr(clip.visual, "image_mean", None)
        self.std = None or getattr(clip.visual, "image_std", None)
        self.image_size = clip.visual.image_size
        self.transform = self.image_transform()
        self.format = format

    def image_transform(self):
        image_size = self.image_size

        mean = self.mean or OPENAI_DATASET_MEAN
        if not isinstance(mean, (list, tuple)):
            mean = (mean,) * 3

        std = self.std or OPENAI_DATASET_STD
        if not isinstance(std, (list, tuple)):
            std = (std,) * 3

        if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
            # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
            image_size = image_size[0]

        transforms = [
            ToPILImage(),
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
        return Compose(transforms)

    def forward(self, image):
        # print(image)
        # print(image.shape)
        # print(image.dtype)
        # print(type(image))
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().astype(np.uint8)
        image = image.astype(np.uint8)
        # print(type(image))
        # if isinstance(image, torch.Tensor):
        #     print("#####")
        #     print(image)
        #     print(image.shape)
        #     print(image.dtype)
        result = []
        if len(image.shape) == 4:
            for x in range(image.shape[0]):
                result.append(self.transform(image[x]))

        else:
            result.append(self.transform(image))

        return torch.stack(result, axis=0).to(self.format)


def main():
    args = parse_args()

    if args.format == "float16":
        format = torch.float16
    elif args.format == "float32":
        format = torch.float32
    elif args.format == "bfloat16":
        format = torch.bfloat16

    # print(open_clip.list_pretrained())
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)

    image_wrapper = ClipImageWrapper(model, format=format).eval()

    if isinstance(model, CustomTextCLIP):
        print("CustomTextCLIP")
        text_wrapper = ClipCustomTextWrapper(model, format=format).eval()
    else:
        print("CLIP")
        text_wrapper = ClipTextWrapper(model, format=format).eval()

    bentoml.pytorch.save_model(
        f"clip_image_encoder",
        image_wrapper,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"{args.model}_{args.pretrained}"},
    )

    bentoml.pytorch.save_model(
        f"clip_text_encoder",
        text_wrapper,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"{args.model}_{args.pretrained}"},
    )

    tokenizer = TokenizerWrapper(open_clip.get_tokenizer(args.model), format=format)

    bentoml.pytorch.save_model(
        f"clip_text_tokenizer",
        tokenizer,
        # signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"{args.model}_{args.pretrained}"},
    )

    image_preprocessor = ImagePreprozessorWrapper(model, format=format)
    bentoml.pytorch.save_model(
        f"clip_image_preprocessor",
        image_preprocessor,
        # signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"{args.model}_{args.pretrained}"},
    )

    ## Test
    if args.test:
        np.random.seed(42)

        image_wrapper.to(args.device)
        text_wrapper.to(args.device)
        start = time.time()

        num_rounds = 20
        for x in range(num_rounds):

            test_image = (np.random.rand(1, 512, 768, 3) * 255).astype(np.uint8)
            image = image_preprocessor(test_image)
            print(image.shape)

            image = image.to(args.device)

            test_text = "An image of a cat"
            text = tokenizer(test_text)
            text = text.to(args.device)
            print(text.shape)

            image_output = image_wrapper(image)
            print(image_output.shape)
            print(image_output[0, :10])

            print(text.device)

            text_output = text_wrapper(text)
            print(text_output.shape)
            print(text_output[0, :10])

            end = time.time()
            print((end - start) / (x + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
