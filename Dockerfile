FROM osgeo/gdal:ubuntu-full-3.6.3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update --fix-missing
RUN apt-get install sudo python3-pip python3-venv git -y
RUN pip3 install rasterio==1.3.8 rio-cogeo --no-binary rasterio
RUN python3 -m pip install -U pip
RUN python3 -m pip install jupyterlab
RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt

RUN useradd -m hanxli
USER hanxli

RUN mkdir /home/hanxli/data
RUN mkdir /home/hanxli/notebook
RUN mkdir /home/hanxli/imageLabeling

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0"]