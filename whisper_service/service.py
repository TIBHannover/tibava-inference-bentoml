import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray

from numpy.typing import NDArray
from typing import Dict, Any


def build_runners():
    return {
        "whisper": bentoml.pytorch.get("whisper:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(text=JSON(), times=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def whisper(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        raw_result = runners["whisper"].run(data)
        print(raw_result)
        return {"text": raw_result.text, "times":np.zeros([])}
