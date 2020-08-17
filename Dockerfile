FROM tensorflow/tensorflow:2.3.0-jupyter

LABEL maintainer="nylakhalil.github.io"

WORKDIR /tf

ENV TFHUB_CACHE_DIR=/tf/object_detection/model/
