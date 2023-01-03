import numpy as np
import bentoml
from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "shot_type_classification": bentoml.torchscript.get("shot_type_classification:latest").to_runner(),
    }


class ShotTypeInput(BaseModel):
    data: List[List[List[List[float]]]]


input_spec = JSON(pydantic_model=ShotTypeInput)


class ShotTypeOutput(BaseModel):
    prob: List[List[float]]


output_spec = JSON(pydantic_model=ShotTypeOutput)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def shot_type_classification(input: ShotTypeInput) -> ShotTypeOutput:
        data = np.asarray(input.data)
        raw_result = runners["shot_type_classification"].run(data)
        return ShotTypeOutput(
            prob=raw_result.cpu().numpy().tolist(),
        )
