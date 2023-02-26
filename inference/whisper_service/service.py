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
        "whisper": bentoml.pytorch.get("whisper:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    def whisper(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        raw_result = runners["whisper"].run(data)
        print(raw_result)
        return {"text": raw_result.text, "times": numpy_to_dict(np.zeros([]))}
