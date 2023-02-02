import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart


from typing import List, Dict, Any
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "places_classification": bentoml.pytorch.get("places_classification:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(embedding=NumpyNdarray(), prob=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    async def places_classification(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        raw_result = await runners["places_classification"].async_run(data)
        return {
            "embedding": raw_result[0].cpu().numpy(),
            "prob": raw_result[1].cpu().numpy(),
        }
