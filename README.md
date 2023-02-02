### oggora

Playing around with TF samples

#### Requirements

- Docker 
- [TF 2.11.0 Docker image](https://hub.docker.com/r/tensorflow/tensorflow)


#### Run

Review [TF GPU requirements](https://www.tensorflow.org/install/pip#hardware_requirements) to ensure GPU compatibility. To levarge GPUs within containers, refer to [TF Docker requirements](https://www.tensorflow.org/install/docker#tensorflow_docker_requirements) and [NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/index.html). 

Tested on Ubuntu 22.10 with NVIDIA GPU, [NVIDIA driver](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

The GPU container runs `nvidia-smi` to verify GPU resource availability.

```sh
# nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.86.01    Driver Version: 515.86.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
                          ..........................
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   3587968      C   /usr/bin/python3                 6959MiB |
+-----------------------------------------------------------------------------+
```

For CPU only usage, remove GPU configs.

##### Start via Compose
```sh
# Build and start container
COMPOSE_PROFILES=gpu,tf docker compose up --build
```

##### Start via CLI

```sh
# Test CUDA container
docker run --name oggora-gpu --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu22.04 nvidia-smi

# Build image
docker image build . --tag oggora-tf:latest

# Run with GPU devices in Docker >= 19.03 
docker run --name oggora-tf --gpus all -it --rm -v $PWD/tf_cache:/tf/cache -v $PWD/object_detection:/tf/object_detection -p 8888:8888 oggora-tf:latest
```

The Dockerfile configures `TFHUB_CACHE_DIR` environment variable to `/tf/cache` which is accessible from repo `tf_cache` directory.

####  Object Detection
Use the `oggora.ipynb` notebook via CLI link or run `/tf/object_detection/oggora.py` within the container.

```
# Directories

object_detection/images/     # Input images 
object_detection/model/      # Model data/download folder
object_detection/oggora.py   # Detection Py
object_detection/output/     # Output images 
```

```sh
# Exec into container to run
python3 /tf/object_detection/oggora.py 
```

![Output image](./object_detection/output/pixel_detect.png?raw=true "Object detection output image of my cat")

####  Image Segmentation

Use the `segmentation.ipynb` notebook via CLI link.

```
# Directories

image_segmentation/images/     # Input images 
image_segmentation/model/      # Model folder
image_segmentation/output/     # Output images 
```

![Output image](./image_segmentation/output/pixel_segment.png?raw=true "Image Segmentation images of my cat")



