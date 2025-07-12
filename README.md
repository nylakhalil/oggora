### oggora

Playing around with PyTorch and TensorFlow examples

#### Requirements

- Docker
- CUDA


#### Run

Review [TF GPU requirements](https://www.tensorflow.org/install/pip#hardware_requirements) to ensure GPU compatibility. To levarge GPUs within containers, refer to [TF Docker requirements](https://www.tensorflow.org/install/docker#tensorflow_docker_requirements) and [NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/index.html). 

Tested on Ubuntu 24.04.1 with NVIDIA GPU, [NVIDIA driver](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

The GPU container runs `nvidia-smi` to verify GPU resource availability. All notebooks tested on a 8GB RTX 3070 Ti GPU.

```sh
# nvidia-smi     
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 Ti     Off |   00000000:01:00.0 Off |                  N/A |
|  0%   39C    P8              5W /  310W |     318MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
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

##### Exec into container 

```sh
docker exec -it oggora-pt /bin/bash
```

### PyTorch (v2.7.1)

####  MiDaS

Use the [midas.ipynb](./pytorch/share/midas/midas.ipynb) notebook for [MiDaS](https://github.com/isl-org/MiDaS).

#### Segment Anything

Use the [sam.ipynb](./pytorch/share/sam/sam.ipynb) notebook for [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything/) from Meta AI FAIR.

#### Stable Diffusion

Use the [diffusers.ipynb](./pytorch/share/diffusers/diffusers.ipynb) notebook for Stable Diffusion via Hugging Face's [diffusers](https://github.com/huggingface/diffusers). The model was taken down by Runway on Hugging Face. Update the URL to download from another repo on Hugging Face.

### TensorFlow (v2.17.0)

#### Object Detection
Use the [oggora.ipynb](tensorflow/share/object_detection/oggora.ipynb) notebook via CLI link or run `/tf/object_detection/oggora.py` within the container.

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

![Output image](./tensorflow/share/object_detection/output/pixel_detect.png?raw=true "Object detection output image of Pixel the cat")

#### Image Segmentation

Use the [segmentation.ipynb](tensorflow/share/image_segmentation/segmentation.ipynb) notebook via CLI link.

```
# Directories

tensorflow/image_segmentation/images/     # Input images 
tensorflow/image_segmentation/model/      # Model folder
tensorflow/image_segmentation/output/     # Output images 
```

![Output image](./tensorflow/share/image_segmentation/output/pixel_segment.png?raw=true "Image Segmentation output image of Pixel the cat")



