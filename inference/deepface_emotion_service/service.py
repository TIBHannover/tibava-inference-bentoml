import bentoml
from bentoml.io import JSON

from typing import Dict

from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "deepface_emotion": bentoml.onnx.get("deepface_emotion:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def deepface_emotion(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        # data = np.asarray(input.data)
        raw_result = await runners["deepface_emotion"].run.async_run(data)

        return {"emotion": numpy_to_dict(raw_result)}
