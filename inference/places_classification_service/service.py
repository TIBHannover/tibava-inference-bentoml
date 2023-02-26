import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart


from typing import List, Dict, Any
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from pydantic import BaseModel
from typing import List
from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "places_classification": bentoml.pytorch.get("places_classification:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def places_classification(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        # data = np.asarray(input.data)
        raw_result = await runners["places_classification"].async_run(data)
        return {
            "embedding": numpy_to_dict(raw_result[0].cpu().numpy()),
            "prob": numpy_to_dict(raw_result[1].cpu().numpy()),
        }
