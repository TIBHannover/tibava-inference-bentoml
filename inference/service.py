import numpy as np
import bentoml
import importlib

services_files = [
    "inference.clip_service.service",
    "inference.shot_type_classification_service.service",
    "inference.transnet_shotdetection_service.service",
    "inference.deepface_emotion_service.service",
    "inference.insightface_detector_service.service",
    "inference.insightface_feature_extraction_service.service",
    "inference.insightface_genderage_service.service",
    "inference.places_classification_service.service",
    "inference.xclip_classification_service.service",
    "inference.whisper_service.service",
]

runners = {}

for service in services_files:
    a = importlib.import_module(service)
    # print(a)
    function_dir = dir(a)
    if "build_runners" in function_dir:
        runners.update(a.build_runners())

svc = bentoml.Service(
    "tibava",
    runners=runners.values(),
)


for service in services_files:
    a = importlib.import_module(service)
    # print(a)
    function_dir = dir(a)
    if "build_apis" in function_dir:
        a.build_apis(svc, runners)


# build_runners
# build_apis
