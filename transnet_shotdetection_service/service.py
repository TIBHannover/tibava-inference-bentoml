import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

from typing import List
from pydantic import BaseModel

from numpy.typing import NDArray


def build_runners():
    return {
        "transnet": bentoml.torchscript.get("transnet:latest").to_runner(),
    }


class TransnetInput(BaseModel):
    data: List[List[List[List[List[float]]]]]


input_spec = JSON(pydantic_model=TransnetInput)


class TransnetOutput(BaseModel):
    single_frame_pred: List[List[List[float]]]
    all_frames_pred: List[List[List[float]]]


output_spec = JSON(pydantic_model=TransnetOutput)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def transnet(input: TransnetInput) -> TransnetOutput:
        data = np.asarray(input.data)
        result = runners["transnet"].run(data)
        return TransnetOutput(
            single_frame_pred=result[0].cpu().numpy().tolist(), all_frames_pred=result[1].cpu().numpy().tolist()
        )
