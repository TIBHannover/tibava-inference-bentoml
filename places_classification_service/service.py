import numpy as np
import bentoml
from bentoml.io import JSON

from typing import TYPE_CHECKING

from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "places_classification": bentoml.pytorch.get("places_classification:latest").to_runner(),
    }


class Places365Input(BaseModel):
    data: List[List[List[List[float]]]]


input_spec = JSON(pydantic_model=Places365Input)


class Places365Output(BaseModel):
    embedding: List[List[float]]
    prob: List[List[float]]


output_spec = JSON(pydantic_model=Places365Output)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def places_classification(input: Places365Input) -> Places365Output:
        data = np.asarray(input.data)
        raw_result = runners["places_classification"].run(data)
        return Places365Output(
            embedding=raw_result[0].cpu().numpy().tolist(),
            prob=raw_result[1].cpu().numpy().tolist(),
        )
