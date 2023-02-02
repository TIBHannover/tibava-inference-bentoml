import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from bentoml.io import JSON

from typing import List, Dict, Any
from pydantic import BaseModel

from numpy.typing import NDArray


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "deepface_emotion": bentoml.onnx.get("deepface_emotion:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(emotion=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    async def deepface_emotion(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        raw_result = await runners["deepface_emotion"].run.async_run(data)

        return {"emotion": raw_result}
