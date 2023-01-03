import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "clip_image_encoder": bentoml.pytorch.get("clip_image_encoder:latest").to_runner(),
        "clip_text_encoder": bentoml.pytorch.get("clip_text_encoder:latest").to_runner(),
        "clip_text_tokenizer": bentoml.pytorch.get("clip_text_tokenizer:latest").to_runner(),
        "clip_image_preprocessor": bentoml.pytorch.get("clip_image_preprocessor:latest").to_runner(),
    }


class CLIPImageInput(BaseModel):
    data: List[List[List[List[float]]]]


image_input_spec = JSON(pydantic_model=CLIPImageInput)


class CLIPImageOutput(BaseModel):
    embedding: List[List[float]]


image_output_spec = JSON(pydantic_model=CLIPImageOutput)


class CLIPTextInput(BaseModel):
    data: str


text_input_spec = JSON(pydantic_model=CLIPTextInput)


class CLIPTextOutput(BaseModel):
    embedding: List[List[float]]


text_output_spec = JSON(pydantic_model=CLIPTextOutput)


def build_apis(service, runners):
    @service.api(input=image_input_spec, output=image_output_spec)
    def clip_image_encoder(input: CLIPImageInput) -> CLIPImageOutput:
        data = np.asarray(input.data)
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
        result = runners["clip_image_preprocessor"].run(data)
        result = runners["clip_image_encoder"].run(result)
        print(result.shape)
        return CLIPImageOutput(embedding=result.cpu().numpy().tolist())

    @service.api(input=text_input_spec, output=text_output_spec)
    def clip_text_encoder(input: CLIPTextInput) -> CLIPTextOutput:

        result = runners["clip_text_tokenizer"].run(input.data)
        result = runners["clip_text_encoder"].run(result)
        return CLIPTextOutput(embedding=result.cpu().numpy().tolist())

    # @service.api(input=Text(), output=NumpyNdarray())
    # def clip_text_tokenizer(input_series: str) -> np.ndarray:
    #     result = runners["clip_text_tokenizer"].run(input_series)
    #     return result

    # @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    # def clip_image_preprocessor(input_series: np.ndarray) -> np.ndarray:
    #     result = runners["clip_image_preprocessor"].run(input_series)
    #     return result
