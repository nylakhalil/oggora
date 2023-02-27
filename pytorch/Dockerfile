
ARG IMAGE_TYPE=devel
ARG CUDA_VERSION=11.3.0 #12.0.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-${IMAGE_TYPE}-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV CUDA_HOME=/usr/local/cuda-11.3
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

RUN pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt 
    
RUN pip install jupyterlab matplotlib

COPY init.sh /tmp/init.sh

WORKDIR /pt

EXPOSE 8888

CMD ["bash", "/tmp/init.sh"]
