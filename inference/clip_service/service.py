import bentoml
from bentoml.io import JSON

from typing import Dict
from inference.utils import dict_to_numpy, numpy_to_dict


def build_runners():
    return {
        "clip_image_encoder": bentoml.pytorch.get("clip_image_encoder:latest").to_runner(),
        "clip_text_encoder": bentoml.pytorch.get("clip_text_encoder:latest").to_runner(),
        "clip_text_tokenizer": bentoml.pytorch.get("clip_text_tokenizer:latest").to_runner(),
        "clip_image_preprocessor": bentoml.pytorch.get("clip_image_preprocessor:latest").to_runner(),
    }


def build_apis(service, runners):
    @service.api(input=JSON(), output=JSON())
    async def clip_image_encoder(input: Dict) -> Dict:
        data = dict_to_numpy(input.get("data"))
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
        result = await runners["clip_image_preprocessor"].async_run(data)
        result = await runners["clip_image_encoder"].async_run(result)
        print(result.shape)
        return {"embedding": numpy_to_dict(result.cpu().numpy())}

    @service.api(input=JSON(), output=JSON())
    async def clip_text_encoder(input: Dict) -> Dict:
        data = input.get("data")
        print(data, flush=True)
        result = await runners["clip_text_tokenizer"].async_run(data)
        result = await runners["clip_text_encoder"].async_run(result)
        return {"embedding": numpy_to_dict(result.cpu().numpy())}

    # @service.api(input=Text(), output=NumpyNdarray())
    # def clip_text_tokenizer(input_series: str) -> np.ndarray:
    #     result = runners["clip_text_tokenizer"].run(input_series)
    #     return result

    # @service.api(input=NumpyNdarray(), output=NumpyNdarray())
    # def clip_image_preprocessor(input_series: np.ndarray) -> np.ndarray:
    #     result = runners["clip_image_preprocessor"].run(input_series)
    #     return result
