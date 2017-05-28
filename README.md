[![Build Status](https://travis-ci.org/ruimashita/dnn.svg?branch=master)](https://travis-ci.org/ruimashita/dnn)
[![codecov](https://codecov.io/gh/ruimashita/dnn/branch/master/graph/badge.svg)](https://codecov.io/gh/ruimashita/dnn)
[![Code Climate](https://codeclimate.com/github/ruimashita/dnn/badges/gpa.svg)](https://codeclimate.com/github/ruimashita/dnn)


# DNN

Automatic differentiation deep neural network framework.
python >= 3.5.


## install
```
$ pip install -U pip setuptools
$ pip install -r /tmp/requirements.txt
$ python setup.py install
```


## example
```
$ python examples/mnist_softmax.py
$ python examples/iris_mlp.py
```


## test

```
$ docker-compose -p dnn run --rm test
```


## build docs

```
$ docker-compose -p dnn build build-docs
$ docker-compose -p dnn run --rm build-docs
```

