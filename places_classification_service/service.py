import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from numpy.typing import NDArray


def build_runners():
    return {
        "places_classification": bentoml.torchscript.get("places_classification:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def places_classification(input_series: np.ndarray) -> np.ndarray:
        result = runners["places_classification"].run(input_series)
        return result
