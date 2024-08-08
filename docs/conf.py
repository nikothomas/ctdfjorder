# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


sys.path.insert(0, os.path.abspath('..'))
project = 'CTDFjorder'
copyright = '2024, Nikolas Yanek-Chrones'
author = 'Nikolas Yanek-Chrones'
release = '0.7.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'pydata_sphinx_theme', 'sphinxarg.ext', 'sphinx_design', 'sphinx.ext.viewcode']
autodoc_default_flags = ['members']
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_toc_level": 2,
    "content_footer_items": ["last-updated"],
    "use_edit_page_button": True,
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_end": [],
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/nikothomas/ctdfjorder",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}
html_css_files = [
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
]

html_logo = "_static/logo.png"
html_context = {
    "github_url": "https://github.com",
    "github_user": "nikothomas",
    "github_repo": "ctdfjorder",
    "github_version": "main",
    "doc_path": "./docs",
}
html_sidebars = {
    "Getting Started": [],
    "researchers": [],
    "developers": [],
}

html_static_path = ['_static']



