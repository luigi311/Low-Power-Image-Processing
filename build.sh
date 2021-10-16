#!/bin/bash

docker buildx build --platform linux/amd64,linux/arm64 -t luigi311/low-power-image-processing:latest --push .
