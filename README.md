### oggora

Playing around with TF samples

#### Requirements

- Docker 
- [TF 2.3.0 Docker image](https://hub.docker.com/r/tensorflow/tensorflow)


#### Setup

```
images/   # Input images 
model/    # Model data/download folder
oggora.py # Detection Py
output/   # Output images 
```

#### Run

Use Docker Compose to start container. 

```
# Build and start container
docker-compose up --build
```

####  Object Detection
Run `/tf/object_detection/oggora.py` within the container or use the `oggora.ipynb` notebook via CLI link.

```
# Exec into container to run
python3 /tf/object_detection/oggora.py 
```






