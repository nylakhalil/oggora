
ARG IMAGE_TYPE=devel
ARG CUDA_VERSION=12.2.2-runtime-ubuntu
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-${IMAGE_TYPE}-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV TORCH_HOME=/pt/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    libgl1 libglib2.0-0 \
    git \
    wget 

RUN ln -s $(which python3) /usr/local/bin/python

RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN pip install jupyterlab matplotlib

COPY init.sh /tmp/init.sh

WORKDIR /pt

EXPOSE 8888

CMD ["bash", "/tmp/init.sh"]
