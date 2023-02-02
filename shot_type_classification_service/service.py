import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray

from numpy.typing import NDArray
from typing import Dict, Any


def build_runners():
    return {
        "shot_type_classification": bentoml.torchscript.get("shot_type_classification:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(prob=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def shot_type_classification(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        raw_result = runners["shot_type_classification"].run(data)
        return {"prob": raw_result.cpu().numpy()}
