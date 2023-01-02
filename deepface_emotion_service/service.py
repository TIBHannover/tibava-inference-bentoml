import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from numpy.typing import NDArray


def build_runners():
    return {
        "deepface_emotion": bentoml.onnx.get("deepface_emotion:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def deepface_emotion(input_series: np.ndarray) -> np.ndarray:
        result = runners["deepface_emotion"].run(input_series)
        return result
