# use pytorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# update the nvidia certificate
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    gcc \
    g++ 

# install opencv requirements
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# clone so-vits-scv repo
RUN git clone https://github.com/svc-develop-team/so-vits-svc.git

# install the requirements file
RUN pip install -r so-vits-svc/requirements.txt

# install libraries
RUN pip install pytube
RUN pip install pydub
RUN pip install youtube-search-python
RUN pip install librosa
RUN pip install samplerate

# copy files
COPY . . 
