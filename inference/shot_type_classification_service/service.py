import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import Multipart
from bentoml.io import NumpyNdarray

from numpy.typing import NDArray
from typing import Dict, Any

from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "shot_type_classification": bentoml.torchscript.get("shot_type_classification:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    def shot_type_classification(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        raw_result = runners["shot_type_classification"].run(data)
        return {"prob": numpy_to_dict(raw_result.cpu().numpy())}
