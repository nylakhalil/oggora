#FROM tensorflow/tensorflow:2.3.0-jupyter
FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

LABEL maintainer="nylakhalil.github.io"

WORKDIR /tf

RUN apt-get update && \
    apt-get install fonts-noto-mono protobuf-compiler -y

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

ENV TFHUB_CACHE_DIR=/tf/object_detection/model/

COPY init.sh /tmp/init.sh
CMD ["bash", "/tmp/init.sh"]
