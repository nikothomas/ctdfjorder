# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.abspath('..'))
font_dirs = ["_static/fonts"]
print(os.getcwd())
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
# Set the font family globally for Matplotlib
plt.rcParams['font.family'] = 'Roboto'
print(plt.rcParams['font.family'])
print(font_manager.findSystemFonts(fontpaths=font_dirs))
project = 'CTDFjorder'
copyright = '2024, Nikolas Yanek-Chrones'
author = 'Nikolas Yanek-Chrones'
release = '0.8.91'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'pydata_sphinx_theme', 'sphinxarg.ext', 'sphinx_design', 'sphinx.ext.viewcode', "sphinx_social_previews"]
ogp_site_url = "https://ctdfjorder.readthedocs.io"
ogp_image = "https://ctdfjorder.readthedocs.io/en/latest/_static/logo.png"
ogp_social_previews = {
    "image_mini": "_static/github-brand.png",
}
ogp_description_length = 500
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
    "show_prev_next": False,
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
    "cli-tutorial": [],
    "researchers": [],
    "developers": [],
    "changelog": []
}

html_static_path = ['_static']


