import numpy as np
import bentoml
from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "insightface_genderage": bentoml.onnx.get("insightface_genderage:latest").to_runner(),
    }


class InsightfaceGenderAgeInput(BaseModel):
    data: List[List[List[List[float]]]]


input_spec = JSON(pydantic_model=InsightfaceGenderAgeInput)
# torch.Size([1, 12800, 1])
# torch.Size([1, 3200, 1])
# torch.Size([1, 800, 1])
# torch.Size([1, 12800, 4])
# torch.Size([1, 3200, 4])
# torch.Size([1, 800, 4])
# torch.Size([1, 12800, 10])
# torch.Size([1, 3200, 10])
# torch.Size([1, 800, 10])


class InsightfaceGenderAgeOutput(BaseModel):
    gender: List[List[float]]
    age: List[List[float]]


output_spec = JSON(pydantic_model=InsightfaceGenderAgeOutput)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def insightface_genderage(input: InsightfaceGenderAgeInput) -> InsightfaceGenderAgeOutput:
        data = np.asarray(input.data)
        raw_result = runners["insightface_genderage"].run.run(data)

        return InsightfaceGenderAgeOutput(gender=raw_result[..., :2].tolist(), age=raw_result[..., 2:].tolist())
