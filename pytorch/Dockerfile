
ARG CUDA_VERSION="12.8.1"
ARG IMAGE_TYPE="-cudnn-runtime-"
ARG IMAGE_OS="ubuntu"
ARG IMAGE_VERSION="24.04"

FROM nvidia/cuda:${CUDA_VERSION}${IMAGE_TYPE}${IMAGE_OS}${IMAGE_VERSION}

ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG="C.UTF-8"
ENV CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
ENV TORCH_HOME="/pt/torch"
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 libglib2.0-0 \
    git \
    wget 

RUN ln -s $(which python3) /usr/local/bin/python

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

RUN /opt/venv/bin/pip3 install jupyterlab matplotlib

COPY init.sh /tmp/init.sh

WORKDIR /pt

EXPOSE 8888

CMD ["bash", "/tmp/init.sh"]
