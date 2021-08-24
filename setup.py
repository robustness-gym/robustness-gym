#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from distutils.util import convert_path
from shutil import rmtree

from setuptools import Command, find_packages, setup

main_ns = {}
ver_path = convert_path("robustnessgym/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


# Package meta-data.
NAME = "robustnessgym"
DESCRIPTION = "Robustness Gym is an evaluation toolkit for machine learning."
URL = "https://github.com/robustness-gym/robustness-gym"
EMAIL = "kgoel@cs.stanford.edu"
AUTHOR = "The Robustness Gym Team"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = main_ns["__version__"]

# What packages are required for this module to be executed?
REQUIRED = [
    "dataclasses>=0.6",
    "pandas",
    "meerkat-ml==0.1.2",
    "dill>=0.3.3",
    "numpy>=1.18.0",
    "cytoolz",
    "ujson",
    "transformers",
    "fastBPE>=0.1.0",
    "jsonlines>=1.2.0",
    "pyahocorasick>-1.4.0",
    "python-Levenshtein>=0.12.0",
    "torch>=1.8.0",
    "tqdm>=4.49.0",
    "datasets>=1.4.1",
    "PyYAML>=5.4.1",
    "omegaconf>=2.0.5",
    "fuzzywuzzy>=0.18.0",
    "semver>=2.13.0",
    "multiprocess>=0.70.11",
    "Cython>=0.29.21",
    "progressbar>=2.5",
    "plotly>=4.14.1",
    "kaleido>=0.1.0",
    "pytorch-lightning>=1.1.2",
    "sklearn",
]

# What packages are optional?
EXTRAS = {
    "dev": [
        "black>=21.5b0",
        "isort>=5.7.0",
        "flake8>=3.8.4",
        "docformatter>=1.4",
        "pytest-cov>=2.10.1",
        "sphinx-rtd-theme>=0.5.1",
        "nbsphinx>=0.8.0",
        "recommonmark>=0.7.1",
        "pre-commit>=2.9.3",
        "sphinx-autobuild",
        "twine",
    ],
    "augmentation": [
        "nlpaug>=1.1.1",
    ],
    "adversarial": [
        "textattack",
    ],
    "text": [
        "nltk",
        "textblob",
        "spacy",
        "stanza",
        "allennlp",
        "allennlp-models",
    ],
    "jupyter": [
        "meerkat-ml==0.1.2[jupyter]",
    ],
    "interactive": ["meerkat-ml==0.1.2[interactive]"],
    "summarization": [
        "rouge-score>=0.0.4",
    ],
    "ludwig": [
        "ludwig",
        "tensorflow",
    ],
    "tabular": [
        "meerkat-ml==0.1.2[tabular]",
    ],
    "vision": [
        "meerkat-ml==0.1.2[vision]",
    ],
}
EXTRAS["all"] = [e for l in EXTRAS.values() for e in l]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
