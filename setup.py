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
    version="",
    author="",
    author_email="kgoel@cs.stanford.edu",
    license="",
    description="Robustness Gym is a toolkit for evaluating the robustness of NLP models.",
    keywords="nlp ml ai deep learning evaluation robustness",
    # url="http://packages.python.org/an_example_pypi_project",
    packages=['robustnessgym'],
    long_description=read('README.md'),
)
