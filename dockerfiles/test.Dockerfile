FROM buildpack-deps:xenial

MAINTAINER takuya.wakisaka@moldweorp.com

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
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ENV PYTHON_VERSIONS 3.5.2 3.6.1

# alias python=python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install -U pip setuptools
RUN pip install tox

# pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

RUN for version in $PYTHON_VERSIONS; do pyenv install $version; done;

RUN pyenv local $PYTHON_VERSIONS

COPY ./ /home
WORKDIR /home

CMD tox
