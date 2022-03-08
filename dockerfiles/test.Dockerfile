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


ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ENV PYTHON_VERSIONS 3.9.10 3.10.2

# alias python=python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN pip3 install -U pip setuptools
RUN pip3 install tox

# pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

RUN for version in $PYTHON_VERSIONS; do pyenv install $version; done;

RUN pyenv local $PYTHON_VERSIONS

COPY ./ /home
WORKDIR /home

CMD tox
