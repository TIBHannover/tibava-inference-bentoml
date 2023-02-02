import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart
from bentoml.io import JSON

from typing import List, Dict, Any
from pydantic import BaseModel

from numpy.typing import NDArray


def build_runners():
    return {
        "transnet": bentoml.torchscript.get("transnet:latest").to_runner(),
    }


input_spec = Multipart(data=NumpyNdarray())

output_spec = Multipart(single_frame_pred=NumpyNdarray(), all_frames_pred=NumpyNdarray())


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    async def transnet(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray()
        print(data.shape)
        result = await runners["transnet"].async_run(data)
        return {
            "single_frame_pred": result[0].cpu().numpy(),
            "all_frames_pred": result[1].cpu().numpy(),
        }
