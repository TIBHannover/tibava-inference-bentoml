import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray, Image, Text
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from typing import List, Dict, Any

from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "insightface_feature_extraction": bentoml.onnx.get("insightface_feature_extraction:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def insightface_feature_extraction(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        # data = np.asarray(input.data)
        raw_result = await runners["insightface_feature_extraction"].run.async_run(data)
        # print(raw_result.shape)
        return {"embedding": numpy_to_dict(raw_result)}
