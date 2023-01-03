import numpy as np
import bentoml
from bentoml.io import JSON


from pydantic import BaseModel
from typing import List


def build_runners():
    return {
        "insightface_feature_extraction": bentoml.onnx.get("insightface_feature_extraction:latest").to_runner(),
    }


class InsightfaceFeatureExtractionInput(BaseModel):
    data: List[List[List[List[float]]]]


input_spec = JSON(pydantic_model=InsightfaceFeatureExtractionInput)


class InsightfaceFeatureExtractionOutput(BaseModel):
    embedding: List[List[float]]


output_spec = JSON(pydantic_model=InsightfaceFeatureExtractionOutput)


def build_apis(service, runners):
    @service.api(input=input_spec, output=output_spec)
    def insightface_feature_extraction(input: InsightfaceFeatureExtractionInput) -> InsightfaceFeatureExtractionOutput:
        data = np.asarray(input.data)
        raw_result = runners["insightface_feature_extraction"].run.run(data)
        print(raw_result.shape)
        return InsightfaceFeatureExtractionOutput(embedding=raw_result.tolist())
