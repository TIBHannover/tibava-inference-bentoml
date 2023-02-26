import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from bentoml.io import JSON

from typing import List, Dict, Any
from pydantic import BaseModel

from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "transnet": bentoml.torchscript.get("transnet:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def transnet(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        # print(data.shape)
        # print(data.dtype)
        # data = np.asarray(input.get("data"))
        result = await runners["transnet"].async_run(data)
        # print("DONE")
        # print(result)
        single_frame_pred = numpy_to_dict(result[0].cpu().numpy())
        # print("A")
        all_frames_pred = numpy_to_dict(result[1].cpu().numpy())
        # print("B")
        return {
            "single_frame_pred": single_frame_pred,
            "all_frames_pred": all_frames_pred,
        }
