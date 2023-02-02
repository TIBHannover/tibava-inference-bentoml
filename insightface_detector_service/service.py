import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart

from typing import List, Dict, Any
from pydantic import BaseModel

from numpy.typing import NDArray


def build_runners():
    return {
        "insightface_detector": bentoml.torchscript.get("insightface_detector:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(
    score_8=NumpyNdarray(),
    score_16=NumpyNdarray(),
    score_32=NumpyNdarray(),
    bbox_8=NumpyNdarray(),
    bbox_16=NumpyNdarray(),
    bbox_32=NumpyNdarray(),
    kps_8=NumpyNdarray(),
    kps_16=NumpyNdarray(),
    kps_32=NumpyNdarray(),
)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    async def insightface_detector(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        raw_result = await runners["insightface_detector"].async_run(data)

        return {
            "score_8": raw_result[0].cpu().numpy(),
            "score_16": raw_result[1].cpu().numpy(),
            "score_32": raw_result[2].cpu().numpy(),
            "bbox_8": raw_result[3].cpu().numpy(),
            "bbox_16": raw_result[4].cpu().numpy(),
            "bbox_32": raw_result[5].cpu().numpy(),
            "kps_8": raw_result[6].cpu().numpy(),
            "kps_16": raw_result[7].cpu().numpy(),
            "kps_32": raw_result[8].cpu().numpy(),
        }
