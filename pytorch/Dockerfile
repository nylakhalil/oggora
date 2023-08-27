
ARG IMAGE_TYPE=devel
ARG CUDA_VERSION=11.6.2 #12.0.1
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-${IMAGE_TYPE}-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV CUDA_HOME=/usr/local/cuda-11.6
ENV TORCH_HOME=/pt/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    libgl1 libglib2.0-0 \
    git \
    wget 

RUN ln -s $(which python3) /usr/local/bin/python

COPY requirements.txt /tmp/

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt 
    
RUN pip install jupyterlab matplotlib

COPY init.sh /tmp/init.sh

WORKDIR /pt

EXPOSE 8888

CMD ["bash", "/tmp/init.sh"]
