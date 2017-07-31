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

RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash


RUN pyenv install 3.5.2
RUN pyenv install 3.6.1

# alias python=python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install -U pip setuptools

COPY test.requirements.txt /tmp/test.requirements.txt
# COPY develop.requirements.txt /tmp/develop.requirements.txt
# RUN pip install -r /tmp/test.requirements.txt
RUN pip install tox


COPY ./ /home
WORKDIR /home

RUN pyenv local 3.5.2 3.6.1

# CMD pip install -e .
