from setuptools import setup

import revpy

setup(
    name='revpy',
    version=revpy.__version__,
    maintainer='joerg doepfert',
    maintainer_email='joerg.doepfert@flixbus.com',
    packages=['revpy'],
    long_description=open('README.md').read(),
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/flix-tech/RevPy'
)
