#!/bin/sh

echo "Starting Jupyter Notebook"
jupyter lab --notebook-dir=/pt --ip 0.0.0.0 --no-browser --allow-root
