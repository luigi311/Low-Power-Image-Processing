FROM python:3.8-slim AS compile-image
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

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses scikit-build pyyaml

WORKDIR /pytorch

RUN git clone --recursive https://github.com/pytorch/pytorch /pytorch && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0

RUN python setup.py install

RUN git clone https://github.com/pytorch/vision.git /torchvision

WORKDIR /torchvision

RUN python setup.py install

# Copy requirements.txt to the container
COPY --chown=app_user:app_user requirements.txt requirements.txt

RUN pip install -r requirements.txt






FROM python:3.8-slim

ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        wget \
        libopenblas-dev \
        sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv

# Create app_user
RUN useradd -m -d /home/app_user -s /bin/bash app_user
# Enable sudo
RUN echo "app_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /app

COPY --chown=app_user:app_user . /app

USER app_user

RUN ./download_models.sh

# Entrypoint entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]