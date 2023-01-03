import numpy as np
import bentoml
from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "deepface_emotion": bentoml.onnx.get("deepface_emotion:latest").to_runner(),
    }


class DeepfaceEmotionInput(BaseModel):
    data: List[List[List[List[float]]]]


input_spec = JSON(pydantic_model=DeepfaceEmotionInput)


class DeepfaceEmotionOutput(BaseModel):
    emotion: List[List[float]]


output_spec = JSON(pydantic_model=DeepfaceEmotionOutput)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def deepface_emotion(input: DeepfaceEmotionInput) -> DeepfaceEmotionOutput:
        data = np.asarray(input.data)
        raw_result = runners["deepface_emotion"].run.run(data)

        return DeepfaceEmotionOutput(emotion=raw_result.tolist())
