from setuptools import setup

import revpy

setup(
    name='revpy',
    version=revpy.__version__,
    packages=['revpy'],
    long_description=open('README.md').read(),
    test_suite='nose.collector',
    tests_require=['nose'],
)
