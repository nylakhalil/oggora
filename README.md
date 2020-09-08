### oggora

Playing around with TF samples

#### Requirements

- Docker 
- [TF 2.3.0 Docker image](https://hub.docker.com/r/tensorflow/tensorflow)


#### Run

For CPU only usage, use Docker Compose to start container. See [Compose issue 6691](https://github.com/docker/compose/issues/6691) for information on Docker Compose GPU support.

```
# Build and start container
docker-compose up --build
```

Review [TF GPU requirements](https://www.tensorflow.org/install/gpu) to ensure GPUs are compatible. To levarge GPUs within containers, refer to [TF Docker requirements](https://www.tensorflow.org/install/docker#tensorflow_docker_requirements) and [NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/index.html). 


Start via CLI to run with GPUs.

```
# Build image
docker image build . --tag oggora:latest

# Run with GPU devices in Docker >= 19.03 
docker run --gpus all -it --rm -v $PWD/tf_cache:/tf/cache -v $PWD/object_detection:/tf/object_detection -p 8888:8888 oggora:latest
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

```
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



