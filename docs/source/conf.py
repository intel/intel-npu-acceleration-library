#! python
#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import glob
import os
import sys


repo_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)
sys.path.insert(0, os.path.join(repo_root, "intel_npu_acceleration_library"))

project = "Intel® NPU Acceleration Library"
copyright = "2024, Intel Corporation"
author = "Intel Corporation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ["_templates"]
exclude_patterns = []

# Add Breathe extension
extensions = [
    # 'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    "breathe",
    "myst_parser",
]

# autodoc_default_options = {
#     'ignore-module-all': False
# }

source_suffix = [".rst", ".md"]

# Breathe Configuration
breathe_default_project = "Intel® NPU Acceleration Library"
breathe_projects = {"Intel® NPU Acceleration Library": "../xml"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
