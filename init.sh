#!/bin/sh

PROTOBUF_STATUS=`dpkg -s protobuf-compiler | grep Status`

if [ "$PROTOBUF_STATUS" != "Status: install ok installed" ]; then
    echo "Installing protobuf-compiler"
    apt-get update && apt-get install -y protobuf-compiler
fi

echo "Protobuf version: $(protoc --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "TensorFlow devices: $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices())')"

echo "Starting Jupyter Notebook"
jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root