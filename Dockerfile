# Use an NVIDIA CUDA image with CUDA 12.3 as the base (non-deprecated)
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

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
        git \
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
        libopencv-dev \
        libgeos-dev \
        libspatialindex-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and dev packages (already available in Ubuntu 22.04)
RUN apt-get update \
    && apt-get install -y python3 python3-venv python3-dev python3-tk \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install pip for Python
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

# Create a clean virtual environment to avoid system package conflicts
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install geospatial packages that depend on system GDAL
RUN /opt/venv/bin/pip install --no-cache-dir \
    fiona==1.9.4 \
    geopandas==0.14.1 \
    pywavelets==1.5.0

# Create requirements file for remaining packages
RUN echo "tensorboard==2.15.1\n\
seaborn==0.13.1\n\
google-streetview==1.2.9\n\
pandas==2.1.3\n\
matplotlib==3.8.2\n\
tqdm==4.66.1\n\
scikit-learn==1.3.2\n\
scikit-image==0.22.0\n\
scipy==1.11.4\n\
exifread==3.0.0\n\
opencv-python-headless==4.8.1.78\n\
jupyterlab==4.0.9" > /tmp/requirements.txt

# Install remaining Python packages via pip
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch, Torchvision, and Torchaudio for CUDA
RUN /opt/venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create a non-root user
RUN useradd -m -s /bin/bash -G sudo user \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up Jupyter configuration for remote access
RUN mkdir -p /home/user/.jupyter \
    && echo "c.NotebookApp.ip = '0.0.0.0'" > /home/user/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> /home/user/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.allow_root = True" >> /home/user/.jupyter/jupyter_notebook_config.py

# Set ownership and permissions
RUN chown -R user:user /home/user

# Set the working directory to the workspace directory
WORKDIR /workspace
RUN mkdir -p /workspace && chown -R user:user /workspace

# Activate virtual environment in .bashrc for user
RUN echo 'source /opt/venv/bin/activate' >> /home/user/.bashrc

# Switch to the non-root user
USER user

# Set up entry point to activate the virtual environment
CMD ["bash", "-c", "source /opt/venv/bin/activate && exec bash"]
