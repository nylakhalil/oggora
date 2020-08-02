### oggora

Playing around with TF samples

#### Requirements

- TF 1.4.0
- TF Models

#### Setup

```
images/   # Input images 
data/     # Drop label files from object_detection/data
model/    # Model data 
oggora.py # Detection Py
output/   # Output images 
utils/    # Drop label_map_util and visualization_utils files from object_detection/utils
```

#### Run

Use Docker Compose to start container. Run `/notebooks/oggora/oggora.py` within the container or use the `oggora.ipynb` notebook for object detection.

```
# Build and start container
docker-compose up --build
```

```
# Exec into container to run
cd /notebooks/oggora
python oggora.py
```



