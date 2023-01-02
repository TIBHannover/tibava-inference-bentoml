#!/bin/sh

mkdir tmp

echo "clip_service"
apptainer exec bentoml.sif python clip_service/export.py -f float16 -d cuda
echo "transnet_shotdetection_service"
apptainer exec bentoml.sif python transnet_shotdetection_service/convert.py -p $1/transnet_shotdetection/transnet_gpu.pt
echo "shot_type_classification_service"
apptainer exec bentoml.sif python shot_type_classification_service/convert.py -p $1/shot_type_classification/shot_type_classifier_e9-s3199_gpu.pt
echo "deepface_emotion_service"
apptainer exec --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python deepface_emotion_service/convert.py -p $1/deepface_emotion/facial_expression_model.onnx
echo "insightface_detector_service"
apptainer exec --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_detector_service/convert.py -p $1/insightface_detector_torch/scrfd_10g_bnkps_gpu.pth
echo "insightface_genderage_service"
apptainer exec --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_genderage_service/convert.py -p $1/insightface_genderage/genderage.onnx
echo "places_classification_service"
apptainer exec --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python places_classification_service/convert.py -p $1/places_classification/resnet50_places365_gpu.pt
echo "insightface_feature_extraction_service"
apptainer exec --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_feature_extraction_service/convert.py -p $1/insightface_feature_extraction/w600k_r50.onnx




echo "build"
apptainer exec --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python --writable-tmpfs --env TEMP=${pwd}/tmp/ bentoml.sif bentoml build




