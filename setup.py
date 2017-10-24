# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

install_requires = ["numpy >= 1.12", 'scipy>=0.19', 'scikit-learn>=0.18']

setup(
    name='dnn',
    version='0.0.1',
    description='DNN numpy',
    long_description="",
    author="Takuya wakisaka",
    author_email='takuya.wakisaka@moldweorp.com',
    url='https://github.com/ruimashita/dnn',
    license="",
    packages=find_packages(exclude=('tests',)),
    install_requires=install_requires,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', ],
)
