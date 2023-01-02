import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from numpy.typing import NDArray


def build_runners():
    return {
        "insightface_detector": bentoml.torchscript.get("insightface_detector:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def insightface_detector(input_series: np.ndarray) -> np.ndarray:
        result = runners["insightface_detector"].run(input_series)
        return result
