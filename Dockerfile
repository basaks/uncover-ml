FROM python:3.10-slim
MAINTAINER Sudipta Basak <sudipta@tasnix.com>

WORKDIR /usr/src/uncover-ml

RUN apt update && apt upgrade -y
RUN apt-get install -y --no-install-recommends \
        make \
        gcc \
        libc6-dev \
        libopenblas-dev \
        libgdal-dev  \
        libhdf5-dev


RUN apt install git openmpi-bin libopenmpi-dev -y \
    && rm -rf /var/lib/apt/lists/* \
    && alias pip=pip3

RUN pip install -U pip

# RUN ./cubist/makecubist .
# RUN pip install -e .[dev]
