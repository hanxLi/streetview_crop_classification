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

4. With WSL:
    - Make sure you are in the root directory of the repo
    - ```docker build -t roadside_crop_classification .``` 
    - ```docker run -it -p 8888:8888 -v $(pwd):/home/hanxli/ roadside_crop_classification```

5. Through the link or though VCS access the notebooks
