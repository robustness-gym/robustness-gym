# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path


version_path = Path(__file__).parent.parent.parent / 'robustnessgym' / "version.py"
metadata = {}
with open(str(version_path)) as ver_file:
    exec(ver_file.read(), metadata)


sys.path.insert(0, os.path.abspath(""))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = "Robustness Gym"
copyright = "2021 Robustness Gym"
author = "Robustness Gym"

# The full version, including alpha/beta/rc tags
# release = "0.0.0dev"
version = release = metadata["__version__"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.autosummary',
    "sphinx_rtd_theme",
    "nbsphinx",
    "recommonmark",

]
autodoc_typehints = "description"
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
# html_theme = 'pytorch_sphinx_theme'
# html_theme_path = ["../../../pytorch_sphinx_theme"]

# html_theme = 'pt_lightning_sphinx_theme'
# import pt_lightning_sphinx_theme
# html_theme_path = [pt_lightning_sphinx_theme.get_html_theme_path()]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

html_logo = '../logo.png'
html_favicon = '../logo.png'

html_theme_options = {
    'pytorch_project': 'https://pytorchlightning.ai',
    # 'canonical_url': about.__docs_url__,
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
}


# Don't show module names in front of class names.
add_module_names = False

# Sort members by group
autodoc_member_order = "groupwise"

autodoc_default_options = {
    'members': True,
    'methods': True,
    'special-members': '__call__',
    'exclude-members': '_abc_impl',
    'show-inheritance': True,
}
