import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from numpy.typing import NDArray


def build_runners():
    return {
        "insightface_feature_extraction": bentoml.onnx.get("insightface_feature_extraction:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def insightface_feature_extraction(input_series: np.ndarray) -> np.ndarray:
        result = runners["insightface_feature_extraction"].run(input_series)
        return result
