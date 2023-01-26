import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "x_clip_sim": bentoml.onnx.get("x_clip_sim:latest").to_runner(),
        "x_clip_text": bentoml.onnx.get("x_clip_text:latest").to_runner(),
        "x_clip_video": bentoml.onnx.get("x_clip_video:latest").to_runner(),
    }


class XCLIPVideoInput(BaseModel):
    data: List[List[List[List[List[float]]]]]


image_input_spec = JSON(pydantic_model=XCLIPVideoInput)


class XCLIPVideoOutput(BaseModel):
    video_features: List[List[float]]
    image_features: List[List[List[float]]]


image_output_spec = JSON(pydantic_model=XCLIPVideoOutput)


class XCLIPTextInput(BaseModel):
    text: List[List[float]]


text_input_spec = JSON(pydantic_model=XCLIPTextInput)


class XCLIPTextOutput(BaseModel):
    text_features: List[List[float]]


text_output_spec = JSON(pydantic_model=XCLIPTextOutput)


class XCLIPSimInput(BaseModel):
    text_features: List[List[float]]
    video_features: List[List[float]]
    image_features: List[List[List[float]]]


sim_input_spec = JSON(pydantic_model=XCLIPSimInput)


class XCLIPSimOutput(BaseModel):
    probs: List[List[float]]
    scale: float


sim_output_spec = JSON(pydantic_model=XCLIPSimOutput)


def build_apis(service, runners):
    @service.api(input=image_input_spec, output=image_output_spec)
    def x_clip_video(input: XCLIPVideoInput) -> XCLIPVideoOutput:
        data = np.asarray(input.data)
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
        result = runners["x_clip_video"].run.run(data)
        # print(result.shape)
        return XCLIPVideoOutput(
            video_features=result[0].tolist(),
            image_features=result[1].tolist(),
        )

    @service.api(input=text_input_spec, output=text_output_spec)
    def x_clip_text(input: XCLIPTextInput) -> XCLIPTextOutput:
        data = np.asarray(input.text)

        result = runners["x_clip_text"].run.run(data)
        print(result.tolist(), flush=True)
        return XCLIPTextOutput(text_features=result.tolist())

    @service.api(input=sim_input_spec, output=sim_output_spec)
    def x_clip_sim(input: XCLIPSimInput) -> XCLIPSimOutput:

        text_features = np.asarray(input.text_features)
        video_features = np.asarray(input.video_features)
        image_features = np.asarray(input.image_features)

        result = runners["x_clip_sim"].run.run(text_features, video_features, image_features)
        return XCLIPSimOutput(
            probs=result[0].tolist(),
            scale=result[1].tolist(),
        )
