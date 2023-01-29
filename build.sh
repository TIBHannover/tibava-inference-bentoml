#!/bin/sh

mkdir -p tmp


echo "insightface_feature_extraction_service"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_feature_extraction_service/convert.py -p $1/insightface_feature_extraction/w600k_r50.onnx

echo "deepface_emotion_service"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python deepface_emotion_service/convert.py -p $1/deepface_emotion/facial_expression_model.onnx

echo "insightface_genderage_service"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_genderage_service/convert.py -p $1/insightface_genderage/genderage.onnx

echo "clip_service"
apptainer exec --bind $2:$HOME/bentoml bentoml.sif python clip_service/export.py

echo "transnet_shotdetection_service"
apptainer exec --bind $2:$HOME/bentoml bentoml.sif python transnet_shotdetection_service/convert.py -p $1/transnet_shotdetection/transnet.pt 

echo "shot_type_classification_service"
apptainer exec --bind $2:$HOME/bentoml bentoml.sif python shot_type_classification_service/convert.py -p $1/shot_type_classification/shot_type_classifier_e9-s3199_gpu.pt

echo "insightface_detector_service"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python insightface_detector_service/convert.py -p $1/insightface_detector_torch/scrfd_10g_bnkps.pth

echo "places_classification_service"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python places_classification_service/export.py

echo "xclip_classification_service"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python bentoml.sif python xclip_classification_service/convert.py --sim_path  $1/xclip/xclip_16_8_sim.onnx --text_path  $1/xclip/xclip_16_8_text.onnx --video_path  $1/xclip/xclip_16_8_video.onnx 

echo "build"
apptainer exec --bind $2:$HOME/bentoml --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python --writable-tmpfs --env TEMP=${pwd}/tmp/ bentoml.sif bentoml build




