import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart

from typing import List, Dict, Any

from bentoml.io import JSON

from numpy.typing import NDArray

from pydantic import BaseModel
from typing import List

from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "x_clip_sim": bentoml.onnx.get("x_clip_sim:latest").to_runner(),
        "x_clip_text": bentoml.onnx.get("x_clip_text:latest").to_runner(),
        "x_clip_video": bentoml.onnx.get("x_clip_video:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def x_clip_video(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        # data = np.asarray(input.data)
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
        result = await runners["x_clip_video"].run.async_run(data)
        # print(result.shape)
        return {"video_features": numpy_to_dict(result[0]), "image_features": numpy_to_dict(result[1])}

    @service.api(input=JSON(), output=JSON())
    async def x_clip_text(input: Dict) -> Dict:
        text = dict_to_numpy(input.get("text"))
        result = await runners["x_clip_text"].run.async_run(text)
        return {"text_features": numpy_to_dict(result)}

    @service.api(input=JSON(), output=JSON())
    async def x_clip_sim(input: Dict) -> Dict:
        text_features = dict_to_numpy(input.get("text_features"))
        video_features = dict_to_numpy(input.get("video_features"))
        image_features = dict_to_numpy(input.get("image_features"))

        result = await runners["x_clip_sim"].run.async_run(text_features, video_features, image_features)
        return {
            "probs": numpy_to_dict(result[0]),
            "scale": numpy_to_dict(result[1]),
        }
