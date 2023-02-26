FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt update --fix-missing -y
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y 
RUN DEBIAN_FRONTEND=noninteractive apt install cmake build-essential protobuf-compiler libprotobuf-dev protobuf-compiler -y 
RUN pip install bentoml[grpc,io-json]
RUN pip install open_clip_torch
RUN pip install -U openai-whisper
RUN pip install transformers[torch]
RUN CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" pip install onnxruntime-gpu onnx protobuf==4.21.11
RUN pip install imageio

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"


CMD ["bentoml", "serve", "tibava:latest", "--production"]