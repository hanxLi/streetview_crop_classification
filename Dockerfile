# Use an NVIDIA CUDA image with CUDA 12.1 as the base
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"
ENV LANG=C.UTF-8

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Install system-level dependencies and add PPA for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        curl \
        sudo \
        gnupg2 \
        vim \
        tmux \
        nano \
        htop \
        wget \
        bash-completion \
        guvcview \
        ffmpeg \
        libsm6 \
        libxext6 \
    && add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        python3-gdal \
        python3-opencv \
        python3-sklearn \
        python3-scipy \
        libopencv-dev \
        libgeos-dev \
        libspatialindex-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and set it as the default version
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-tk \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install Fiona and other Python packages via pip
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install \
        fiona \
        tensorboard \
        seaborn \
        google-streetview \
        pandas \
        geopandas \
        matplotlib \
        tqdm \
        scikit-learn \
        scikit-image \
        scipy \
        exifread \
        opencv-python-headless \
        GDAL \
        jupyterlab

# Install PyTorch, Torchvision, and Torchaudio for CUDA 12.1
RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set the working directory to the home directory of the user
WORKDIR /home/user
