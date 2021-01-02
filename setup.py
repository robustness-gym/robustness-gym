"""
Setup script for the Robustness Gym library.
"""
import os
from setuptools import setup, find_packages


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Robustness Gym",
    version="0.0.1",
    author="Stanford Hazy Research, Salesforce Research",
    author_email="kgoel@cs.stanford.edu",
    license="Apache 2.0",
    description="Robustness Gym is an evaluation toolkit for natural language processing.",
    keywords="nlp ml ai deep learning evaluation robustness",
    url="https://github.com/robustness-gym/robustness-gym",
    packages=['robustnessgym'],
    long_description=read('README.md'),
)
