FROM buildpack-deps:xenial

MAINTAINER takuya.wakisaka@moldweorp.com

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# alias python=python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install -U pip setuptools

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ./ /home
WORKDIR /home
RUN python setup.py install
