### oggora

Playing around with PyTorch and TensorFlow examples

#### Requirements

- Docker
- CUDA


#### Run

Review [TF GPU requirements](https://www.tensorflow.org/install/pip#hardware_requirements) to ensure GPU compatibility. To levarge GPUs within containers, refer to [TF Docker requirements](https://www.tensorflow.org/install/docker#tensorflow_docker_requirements) and [NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/index.html). 

Tested on Ubuntu 22.10 with NVIDIA GPU, [NVIDIA driver](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

The GPU container runs `nvidia-smi` to verify GPU resource availability.

```sh
# nvidia-smi
+-----------------------------------------------------------------------------+
|  NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0    |
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

##### Start via Compose

```sh
# Build and start TensorFlow container
COMPOSE_PROFILES=gpu,tf docker compose up --build
```

```sh
# Build and start PyTorch container
COMPOSE_PROFILES=gpu,pt docker compose up --build
```

##### Start via CLI

```sh
# Test CUDA container
docker run --name oggora-gpu --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Build image
docker image build . --tag oggora-tf:latest

# Run with GPU devices in Docker >= 19.03 
docker run --name oggora-tf --gpus all -it --rm -v $PWD/tf_cache:/tf/cache -v $PWD/object_detection:/tf/object_detection -p 8888:8888 oggora-tf:latest
```

The Dockerfile configures `TFHUB_CACHE_DIR` environment variable to `/tf/cache` which is accessible from repo `tf_cache` directory.

### PyTorch

Use the `segmentation.ipynb` notebook via CLI link to use [Mask2Former](https://arxiv.org/abs/2112.01527) in [Detectron2](https://github.com/facebookresearch/detectron2).

```
# Directories

pytorch/segmentation/input/       # Input images
pytorch/segmentation/Mask2Former  # Model folder
pytorch/segmentation/output/      # Output images 
```

![Output image](./pytorch/segmentation/output/combined_result_pixel.jpg?raw=true "Segmentation output image of Pixel the cat")


### TensorFlow

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

![Output image](./tensorflow/object_detection/output/pixel_detect.png?raw=true "Object detection output image of Pixel the cat")

####  Image Segmentation

Use the `segmentation.ipynb` notebook via CLI link.

```
# Directories

tensorflow/image_segmentation/images/     # Input images 
tensorflow/image_segmentation/model/      # Model folder
tensorflow/image_segmentation/output/     # Output images 
```

![Output image](./tensorflow/image_segmentation/output/pixel_segment.png?raw=true "Image Segmentation output image of Pixel the cat")



