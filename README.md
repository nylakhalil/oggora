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
docker run --gpus all -it --rm -v object_detection:/tf/object_detection -p 8888:8888 oggora:latest
```

####  Object Detection
Run `/tf/object_detection/oggora.py` within the container or use the `oggora.ipynb` notebook via CLI link.

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






