import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text
from bentoml.io import Multipart

from bentoml.io import JSON

from typing import List, Dict, Any
from pydantic import BaseModel

from numpy.typing import NDArray


def build_runners():
    return {
        "clip_image_encoder": bentoml.pytorch.get("clip_image_encoder:latest").to_runner(),
        "clip_text_encoder": bentoml.pytorch.get("clip_text_encoder:latest").to_runner(),
        "clip_text_tokenizer": bentoml.pytorch.get("clip_text_tokenizer:latest").to_runner(),
        "clip_image_preprocessor": bentoml.pytorch.get("clip_image_preprocessor:latest").to_runner(),
    }


image_input_spec = Multipart(data=NumpyNdarray())

image_output_spec = Multipart(embedding=NumpyNdarray())


text_input_spec = Multipart(data=Text())

text_output_spec = Multipart(embedding=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=image_input_spec, output=image_output_spec)
    async def clip_image_encoder(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:

        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
        result = await runners["clip_image_preprocessor"].async_run(data)
        result = await runners["clip_image_encoder"].async_run(result)
        print(result.shape)
        return {"embedding": result.cpu().numpy()}

    @service.api(input=text_input_spec, output=text_output_spec)
    async def clip_text_encoder(data: str) -> Dict[str, NDArray[Any]]:
        print(data, flush=True)
        result = await runners["clip_text_tokenizer"].async_run(data)
        result = await runners["clip_text_encoder"].async_run(result)
        return {"embedding": result.cpu().numpy()}

    # @service.api(input=Text(), output=NumpyNdarray())
    # def clip_text_tokenizer(input_series: str) -> np.ndarray:
    #     result = runners["clip_text_tokenizer"].run(input_series)
    #     return result

    # @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    # def clip_image_preprocessor(input_series: np.ndarray) -> np.ndarray:
    #     result = runners["clip_image_preprocessor"].run(input_series)
    #     return result
