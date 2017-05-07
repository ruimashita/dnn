FROM buildpack-deps:xenial

MAINTAINER takuya.wakisaka@moldweorp.com

# TensorBoard
EXPOSE 6006

RUN echo "deb http://ftp.jaist.ac.jp/ubuntu/ xenial main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ xenial-updates main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial-updates main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb http://security.ubuntu.com/ubuntu xenial-security main restricted universe multiverse \n\
deb-src http://security.ubuntu.com/ubuntu xenial-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-numpy \
    python3-scipy \
    python3-pil \
    python3-matplotlib \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip

RUN wget https://bootstrap.pypa.io/ez_setup.py -O - | python3

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# develop
RUN apt-get update && apt-get install -y \
    x11-apps \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY develop.requirements.txt /tmp/develop.requirements.txt
# RUN pip3 install --user -r /tmp/develop.requirements.txt
RUN pip3 install -r /tmp/develop.requirements.txt

COPY ./ /home
WORKDIR /home
CMD python3 setup.py install
