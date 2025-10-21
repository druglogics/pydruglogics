# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyDrugLogics'
copyright = '2024, Laura Szekeres'
author = 'Laura Szekeres'
version = '0.1.10'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    # 'myst_parser',
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_rtd_theme",
]

autoclass_content = 'both'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']
autodoc_typehints = "none"

sys.path.insert(0, os.path.abspath('../'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

master_doc = 'index'

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']

def autodoc_skip_docstring(app, what, name, obj, options, lines):
    lines.clear()

def setup(app):
    app.connect("autodoc-process-docstring", autodoc_skip_docstring)

