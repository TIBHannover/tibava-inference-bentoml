import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray, Image, Text
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from typing import List, Dict, Any


from numpy.typing import NDArray
from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "insightface_feature_extraction": bentoml.onnx.get("insightface_feature_extraction:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(embedding=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    async def insightface_feature_extraction(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        raw_result = await runners["insightface_feature_extraction"].run.async_run(data)
        # print(raw_result.shape)
        return {"embedding":raw_result}
