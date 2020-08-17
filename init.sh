#!/bin/sh

PROTOBUF_STATUS=`dpkg -s protobuf-compiler | grep Status`

if [ "$PROTOBUF_STATUS" != "Status: install ok installed" ]; then
    echo "Installing protobuf-compiler"
    apt-get update && apt-get install -y protobuf-compiler
fi

echo "Protobuf version is" $(protoc --version)
echo "TensorFlow version is" $(python -c 'import tensorflow as tf; print(tf.__version__)')

echo "Installing fonts"
apt-get update && apt-get install fonts-noto-mono -y

echo "Installing tensorflow_hub"
pip3 install tensorflow_hub

echo "Starting Jupyter Notebook"
/etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root