[![Build Status](https://travis-ci.org/ruimashita/dnn.svg?branch=master)](https://travis-ci.org/ruimashita/dnn)
[![codecov](https://codecov.io/gh/ruimashita/dnn/branch/master/graph/badge.svg)](https://codecov.io/gh/ruimashita/dnn)
[![Code Climate](https://codeclimate.com/github/ruimashita/dnn/badges/gpa.svg)](https://codeclimate.com/github/ruimashita/dnn)


# DNN

Automatic differentiation deep neural network framework.

## Getting Started

python >= 3.5.

### Installing

Please upgrade pip before installing.
```
$ pip install -U pip setuptools
```

There are two ways to install.

* git clone & install
```
$ git clone https://github.com/ruimashita/dnn.git
$ cd dnn
$ pip install .
```

* install from github
```
$ pip install git+https://github.com/ruimashita/dnn.git
```


### Example
```
$ python examples/mnist_softmax.py
$ python examples/iris_mlp.py
```


## Test

```
$ docker-compose -p dnn build test
$ docker-compose -p dnn run --rm test
```


## Build docs

```
$ docker-compose -p dnn build build-docs
$ docker-compose -p dnn run --rm build-docs
```

