FROM python:3.8-slim

ENV USE_CUDA=0
ENV USE_ROCM=0
ENV USE_NCCL=0
ENV USE_DISTRIBUTED=0
ENV USE_PYTORCH_QNNPACK=0
ENV MAX_JOBS=12

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libopenblas-dev \
        git \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses scikit-build pyyaml

WORKDIR /pytorch

RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0

RUN python setup.py install

RUN git clone https://github.com/pytorch/vision.git /torchvision

WORKDIR /torchvision

RUN python setup.py install

# Create app_user
RUN useradd -m -d /home/app_user -s /bin/bash app_user

WORKDIR /app
# Copy requirements.txt to the container
COPY --chown=app_user:app_user requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY --chown=app_user:app_user . /app

USER app_user

RUN ./download_models.sh