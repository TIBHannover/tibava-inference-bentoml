import logging
from numpy.typing import NDArray
from typing import AnyStr, Union, List, Dict
import base64

import numpy as np


def numpy_to_dict(nd_array: NDArray):
    return {
        "data": base64.b64encode(np.ascontiguousarray(nd_array)).decode("utf-8"),
        "dtype": str(nd_array.dtype),
        "shape": nd_array.shape,
    }


def dict_to_numpy(data: Dict):
    if "data" not in data or "dtype" not in data or "shape" not in data:
        return
    try:
        return np.frombuffer(base64.decodebytes(data["data"].encode()), dtype=data["dtype"]).reshape(data["shape"])
    except Exception as e:
        logging.error(f"[BentoMLInferenceServer] dict_to_numpy {e}")
    return
