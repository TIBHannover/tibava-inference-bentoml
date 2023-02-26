import numpy as np
import bentoml
from bentoml.io import JSON

from typing import List, Dict, Any

from numpy.typing import NDArray


from pydantic import BaseModel
from typing import List

from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "insightface_genderage": bentoml.onnx.get("insightface_genderage:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def insightface_genderage(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        # data = np.asarray(input.data)
        raw_result = await runners["insightface_genderage"].run.run_async(data)

        return {"gender": numpy_to_dict(raw_result[..., :2]), "age": numpy_to_dict(raw_result[..., 2:])}
