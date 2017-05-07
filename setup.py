# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession

requirements = [str(req.req) for req in parse_requirements("requirements.txt", session=PipSession())]

setup(
    name='dnn',
    version='0.0.1',
    description='DNN numpy',
    long_description="",
    author="Takuya wakisaka",
    author_email='takuya.wakisaka@moldweorp.com',
    url='https://github.com/ruimashita/dnn',
    license="",
    packages=find_packages(exclude=('tests', 'deprecated')),
    install_requires=requirements,
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
)
