# Crop type Classification using Streetview-level Images

#### Hanxi Li, GMU, CSISS

### Quick Start

#### Docker

1. [Install Docker](https://docs.docker.com/get-docker/)
    - Follow prompt to install docker
    - For windows make sure WSL2 is installed

2. [Install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    - Follow prompt to install Container Toolkit in docker

3. Clone this repository

4. Building Environment:
    - Make sure you are in the root directory of the repo
      - For MacOS with MPS Support: ```docker build -t street_view_classification -f Dockerfile_MPS .``` 
      - For Linux/WSL with CUDA Support: ```docker build -t street_view_classification_cuda -f Dockerfile_CUDA .```
    - For WSL or Linux with Nvidia GPU and CUDA support:
      - ```docker run -it --rm --gpus all -p 8888:8888 -v "%cd%/data:/workspace/data" -v "%cd%/streetview_crop_classification:/workspace/streetview_crop_classification" --name street_view_classification_container street_view_classification```
    - For MacOS with M-series Chip and MPS support:
      - ``` docker run -it -p 8888:8888 -v "$(pwd)/data:/workspace/data" -v "$(pwd)/streetview_crop_classification:/workspace/streetview_crop_classification" --name street_view_classification_container street_view_classification```
5. Access the notebooks
   - Attach Visual Code Studio to Container, or
   - Open __http://localhost:8888__ once containeris up and running
