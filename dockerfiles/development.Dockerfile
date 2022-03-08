FROM buildpack-deps:focal

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

RUN pip3 install -U pip setuptools

# develop
RUN apt-get update && apt-get install -y \
    x11-apps \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY test.requirements.txt /tmp/test.requirements.txt
COPY develop.requirements.txt /tmp/develop.requirements.txt
RUN pip3 install -r /tmp/develop.requirements.txt

COPY ./ /home
WORKDIR /home
RUN pip3 install -e .
