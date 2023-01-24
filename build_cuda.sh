#!/bin/sh

mkdir tmp


echo "insightface_feature_extraction_service"
apptainer exec --nv --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_feature_extraction_service/convert.py -p $1/insightface_feature_extraction/w600k_r50.onnx -t  -d cuda

echo "deepface_emotion_service"
apptainer exec --nv --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python deepface_emotion_service/convert.py -p $1/deepface_emotion/facial_expression_model.onnx -t  -d cuda

echo "insightface_genderage_service"
apptainer exec --nv --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_genderage_service/convert.py -p $1/insightface_genderage/genderage.onnx -t  -d cuda

echo "clip_service"
apptainer exec --nv --bind $2:$HOME/bentoml bentoml.sif python clip_service/export.py -f float16 -d cuda -t

echo "transnet_shotdetection_service"
apptainer exec --nv --bind $2:$HOME/bentoml bentoml.sif python transnet_shotdetection_service/convert.py -p $1/transnet_shotdetection/transnet_gpu.pt -d cuda -t

echo "shot_type_classification_service"
apptainer exec --nv --bind $2:$HOME/bentoml bentoml.sif python shot_type_classification_service/convert.py -p $1/shot_type_classification/shot_type_classifier_e9-s3199_gpu.pt -t -d cuda

echo "insightface_detector_service"
apptainer exec --nv --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_detector_service/convert.py -p $1/insightface_detector_torch/scrfd_10g_bnkps.pth -t -d cuda 

echo "places_classification_service"
apptainer exec --nv --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python places_classification_service/export.py -f float16 -t -d cuda


echo "build"
apptainer exec --nv --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python --writable-tmpfs --env TEMP=${pwd}/tmp/ bentoml.sif bentoml build




