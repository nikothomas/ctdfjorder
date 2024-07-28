# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sys
sys.setrecursionlimit(50000)

sys.path.insert(0, os.path.abspath('..'))
project = 'CTDFjorder'
copyright = '2024, Nikolas Yanek-Chrones'
author = 'Nikolas Yanek-Chrones'
release = '0.5.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'pydata_sphinx_theme', 'sphinxarg.ext']
autodoc_default_flags = ['members']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_toc_level": 1,
    "content_footer_items": ["last-updated"],
}
html_sidebars = {
    "Getting Started": [],
    "researchers": [],
    "developers": [],
}

html_static_path = ['_static']



