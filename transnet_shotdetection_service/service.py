import numpy as np
import logging
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from bentoml.io import JSON

from typing import List, Dict, Any
from pydantic import BaseModel

from numpy.typing import NDArray

import base64



def numpy_to_dict(nd_array:NDArray):
    return {
        "data": base64.b64encode(nd_array).decode("utf-8"),
        "dtype": str(nd_array.dtype),
        "shape": nd_array.shape,
    }

def dict_to_numpy(data:Dict):
    if "data" not in data or "dtype" not in data or "shape" not in data:
        return 
    # try:
    return np.frombuffer(base64.decodebytes(data["data"].encode()), dtype=data["dtype"]).reshape(data["shape"])
    # except Exception as e:
    #     logging.error(f"[BentoMLInferenceServer] dict_to_numpy {e}")


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
        single_frame_pred=  numpy_to_dict(result[0].cpu().numpy())
        # print("A")
        all_frames_pred=  numpy_to_dict(result[1].cpu().numpy())
        # print("B")
        return {
            "single_frame_pred": single_frame_pred,
            "all_frames_pred": all_frames_pred,
        }
