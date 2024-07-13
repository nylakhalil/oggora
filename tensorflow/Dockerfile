FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-mono protobuf-compiler

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
		tensorflow-hub>=0.16.1 \
		tensorflow_addons>=0.23.0 \
		tensorflow-datasets>=4.9.6

ENV TFHUB_CACHE_DIR=/tf/cache/

COPY init.sh /tmp/init.sh

WORKDIR /tf

EXPOSE 8888

CMD ["bash", "/tmp/init.sh"]

