import numpy as np
import bentoml
from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "insightface_detector": bentoml.torchscript.get("insightface_detector:latest").to_runner(),
    }


class InsightfaceDetectorInput(BaseModel):
    data: List[List[List[List[float]]]]


input_spec = JSON(pydantic_model=InsightfaceDetectorInput)
# torch.Size([1, 12800, 1])
# torch.Size([1, 3200, 1])
# torch.Size([1, 800, 1])
# torch.Size([1, 12800, 4])
# torch.Size([1, 3200, 4])
# torch.Size([1, 800, 4])
# torch.Size([1, 12800, 10])
# torch.Size([1, 3200, 10])
# torch.Size([1, 800, 10])


class InsightfaceDetectorOutput(BaseModel):
    score_8: List[List[List[float]]]
    score_16: List[List[List[float]]]
    score_32: List[List[List[float]]]
    bbox_8: List[List[List[float]]]
    bbox_16: List[List[List[float]]]
    bbox_32: List[List[List[float]]]
    kps_8: List[List[List[float]]]
    kps_16: List[List[List[float]]]
    kps_32: List[List[List[float]]]


output_spec = JSON(pydantic_model=InsightfaceDetectorOutput)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def insightface_detector(input: InsightfaceDetectorInput) -> InsightfaceDetectorOutput:
        data = np.asarray(input.data)
        raw_result = runners["insightface_detector"].run(data)

        return InsightfaceDetectorOutput(
            score_8=raw_result[0].cpu().numpy().tolist(),
            score_16=raw_result[1].cpu().numpy().tolist(),
            score_32=raw_result[2].cpu().numpy().tolist(),
            bbox_8=raw_result[3].cpu().numpy().tolist(),
            bbox_16=raw_result[4].cpu().numpy().tolist(),
            bbox_32=raw_result[5].cpu().numpy().tolist(),
            kps_8=raw_result[6].cpu().numpy().tolist(),
            kps_16=raw_result[7].cpu().numpy().tolist(),
            kps_32=raw_result[8].cpu().numpy().tolist(),
        )
