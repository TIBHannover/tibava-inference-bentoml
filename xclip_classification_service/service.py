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


def build_runners():
    return {
        "x_clip_sim": bentoml.onnx.get("x_clip_sim:latest").to_runner(),
        "x_clip_text": bentoml.onnx.get("x_clip_text:latest").to_runner(),
        "x_clip_video": bentoml.onnx.get("x_clip_video:latest").to_runner(),
    }


image_input_spec = Multipart(data=NumpyNdarray())

image_output_spec = Multipart(video_features=NumpyNdarray(), image_features=NumpyNdarray())

text_input_spec = Multipart(text=NumpyNdarray())

text_output_spec = Multipart(text_features=NumpyNdarray())

sim_input_spec = Multipart(text_features=NumpyNdarray(), video_features=NumpyNdarray(), image_features=NumpyNdarray())

sim_output_spec = Multipart(probs=NumpyNdarray(), scale=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=image_input_spec, output=image_output_spec)
    async def x_clip_video(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
        result = await runners["x_clip_video"].run.async_run(data)
        # print(result.shape)
        return {"video_features": result[0], "image_features": result[1]}

    @service.api(input=text_input_spec, output=text_output_spec)
    async def x_clip_text(text: NDArray[Any]) -> Dict[str, NDArray[Any]]:

        result = await runners["x_clip_text"].run.async_run(text)
        print(result.tolist(), flush=True)
        return {"text_features": result}

    @service.api(input=sim_input_spec, output=sim_output_spec)
    async def x_clip_sim(
        text_features: NDArray, video_features: NDArray, image_features: NDArray
    ) -> Dict[str, NDArray[Any]]:

        result = await runners["x_clip_sim"].run.async_run(text_features, video_features, image_features)
        return {
            "probs": result[0],
            "scale": result[1],
        }
