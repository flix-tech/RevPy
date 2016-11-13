from setuptools import setup

setup(
    name='PyRM2',
    version='0.1',
    packages=['pyrm'],
    long_description=open('README.md').read(),
    test_suite='nose.collector',
    tests_require=['nose'],
)
