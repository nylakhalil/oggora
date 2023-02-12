FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-mono protobuf-compiler

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

ENV TFHUB_CACHE_DIR=/tf/cache/

COPY init.sh /tmp/init.sh

WORKDIR /tf

EXPOSE 8888

CMD ["bash", "/tmp/init.sh"]

