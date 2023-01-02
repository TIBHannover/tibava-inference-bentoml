import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image, Text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
    from numpy.typing import NDArray


def build_runners():
    return {
        "clip_image_encoder": bentoml.pytorch.get("clip_image_encoder:latest").to_runner(),
        "clip_text_encoder": bentoml.pytorch.get("clip_text_encoder:latest").to_runner(),
        "clip_text_tokenizer": bentoml.pytorch.get("clip_text_tokenizer:latest").to_runner(),
        "clip_image_preprocessor": bentoml.pytorch.get("clip_image_preprocessor:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def clip_image_encoder(input_series: np.ndarray) -> np.ndarray:
        result = runners["clip_image_encoder"].run(input_series)
        return result

    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def clip_text_encoder(input_series: np.ndarray) -> np.ndarray:
        result = runners["clip_text_encoder"].run(input_series)
        return result

    @service.api(input=Text(), output=NumpyNdarray())
    def clip_text_tokenizer(input_series: str) -> np.ndarray:
        result = runners["clip_text_tokenizer"].run(input_series)
        return result

    @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    def clip_image_preprocessor(input_series: np.ndarray) -> np.ndarray:
        result = runners["clip_image_preprocessor"].run(input_series)
        return result
