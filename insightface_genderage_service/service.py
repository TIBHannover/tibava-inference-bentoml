import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from bentoml.io import JSON

from typing import List, Dict, Any

from numpy.typing import NDArray


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "insightface_genderage": bentoml.onnx.get("insightface_genderage:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(gender=NumpyNdarray(), age=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    async def insightface_genderage(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        raw_result = await runners["insightface_genderage"].run.run_async(data)

        return {"gender": raw_result[..., :2], "age": raw_result[..., 2:]}
