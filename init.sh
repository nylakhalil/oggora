#!/bin/sh

PROTOBUF_STATUS=`dpkg -s protobuf-compiler | grep Status`

if [ "$PROTOBUF_STATUS" != "Status: install ok installed" ]; then
    echo "Installing protobuf-compiler"
    apt-get update && apt-get install -y protobuf-compiler
fi

echo "Protobuf version is" $(protoc --version)
echo "TensorFlow version is" $(python -c 'import tensorflow as tf; print(tf.__version__)')

protoc --proto_path=/notebooks/oggora/protos /notebooks/oggora/protos/*.proto --python_out=/notebooks/oggora/protos
echo "Compiled protobufs" $(ls /notebooks/oggora/protos)

echo "Starting Jupyter Notebook"
/run_jupyter.sh --allow-root