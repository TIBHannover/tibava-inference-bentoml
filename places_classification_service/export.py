import imageio
import numpy as np
from PIL import Image

from torch.autograd import Variable as V
from torchvision import transforms as trn
import torchvision.models as models
import torch
import bentoml


class PlaceClassifierJIT(torch.nn.Module):
    def __init__(self, model_file: str, arch: str = "resnet50", device="cpu", format=torch.float32) -> None:
        super(PlaceClassifierJIT, self).__init__()

        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.format = format
        self.to(format)

    def preprocess(self, input):

        mean = torch.zeros(3).float().to(input.device)
        std = torch.zeros(3).float().to(input.device)
        mean[0], mean[1], mean[2] = 0.485, 0.456, 0.406
        std[0], std[1], std[2] = 0.229, 0.224, 0.225
        mean = mean.unsqueeze(1).unsqueeze(1)
        std = std.unsqueeze(1).unsqueeze(1)
        temp = input.float().div(255).permute(2, 0, 1).to(input.device)

        return temp.sub(mean).div(std).unsqueeze(0)

    def forward(self, input):

        with torch.no_grad():
            inputs = []
            for i in range(input.shape[0]):
                inputs.append(self.preprocess(input[i, ...]))
            x = torch.concat(inputs, dim=0).to(self.format)

            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)

            x = self.model.fc(features)
            return features, torch.nn.functional.softmax(x, 1)


import os
import sys
import re
import argparse
import urllib.request
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-m", "--model_path", help="verbose output")
    parser.add_argument("-f", "--format", default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("-t", "--test", action="store_true", help="verbose output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.format == "float16":
        format = torch.float16
    elif args.format == "float32":
        format = torch.float32
    elif args.format == "bfloat16":
        format = torch.bfloat16

    model_path = args.model_path
    if model_path is None:
        ...
        url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
        response = urllib.request.urlopen(url)
        tmp_dir = tempfile.mkdtemp()
        model_path = os.path.join(tmp_dir, "resnet50_places365.pth.tar")
        with open(model_path, "wb") as f:
            f.write(response.read())

    # load pytorch model
    PCJIT = PlaceClassifierJIT(model_file=model_path, device=args.device, format=format)
    PCJIT.eval()
    # .to(args.device)

    bentoml.pytorch.save_model(
        f"places_classification",
        PCJIT,
        signatures={"__call__": {"batchable": True, "batch_dim": 0}},
        labels={"model": f"places_classification"},
    )

    if args.test:
        PCJIT.to(args.device)
        np.random.seed(42)
        test_image = (np.random.rand(1, 224, 224, 3) * 255).astype(np.uint8)
        output = PCJIT(torch.as_tensor(test_image).to(args.device))
        if isinstance(output, (list, set, tuple)):
            for x in output:
                print(x.shape)
        else:

            print(output.shape)
    # # forward pass
    # features, preds = PCJIT(input_img)
    # features = features.cpu().detach().numpy()[0]
    # preds = preds.cpu().detach().numpy()[0]

    # print(features.shape)
    # print(features[:10])
    # print(preds.shape)
    # print(preds[:10])

    # """
    # result with new code and jit model
    # """

    # # trace model
    # traced_model = torch.jit.trace(PCJIT, input_img)
    # torch.jit.save(traced_model, args.output_path)

    # features_traced, preds_traced = traced_model(input_img)
    # features_traced = features_traced.cpu().detach().numpy()[0]
    # preds_traced = preds_traced.cpu().detach().numpy()[0]

    # print(features_traced.shape)
    # print(features_traced[:10])
    # print(preds_traced.shape)
    # print(preds_traced[:10])

    # """
    # result comparisons
    # """

    # # make sure new code produces same results than original code
    # np.testing.assert_allclose(preds_org, preds, rtol=1e-3)

    # # make sure that both models produce similar results
    # np.testing.assert_allclose(features, features_traced, rtol=1e-3)
    # np.testing.assert_allclose(preds, preds_traced, rtol=1e-3)

    return 0


if __name__ == "__main__":
    sys.exit(main())
