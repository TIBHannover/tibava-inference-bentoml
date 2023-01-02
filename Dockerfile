FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt update --fix-missing -y
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y 
RUN pip install bentoml[grpc,io-json]
RUN pip install open_clip_torch
RUN pip install transformers[torch]
RUN pip install onnxruntime-gpu onnx protobuf==4.21.11
RUN pip install imageio

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"

CMD ["bentoml", "serve", "tibava:latest", "--production"]