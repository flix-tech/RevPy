from setuptools import setup

import pyrm

setup(
    name='PyRM2',
    version=pyrm.__version__,
    packages=['pyrm'],
    long_description=open('README.md').read(),
    test_suite='nose.collector',
    tests_require=['nose'],
)
