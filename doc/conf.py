# -*- coding: utf-8 -*-
#
# hyperspy-ml-algorithms documentation build configuration file

import os
import sys
from datetime import datetime
from importlib.metadata import version

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration -----------------------------------------------

extensions = [
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]

autosummary_generate = True

source_suffix = ".rst"

master_doc = "index"

project = "hyperspy-ml-algorithms"
copyright = f"2024-{datetime.today().year}, The HyperSpy development team"

release = version("hyperspy-ml-algorithms")
version = ".".join(release.split(".")[:2])

exclude_patterns = ["_build"]

pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/hyperspy/hyperspy-ml-algorithms",
    "show_toc_level": 2,
    "logo": {
        "text": "hyperspy-ml-algorithms",
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# -- Intersphinx ---------------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current", None),
}

# -- Options for numpydoc ------------------------------------------------

numpydoc_show_class_members = False

# -- Autodoc / Autosummary -----------------------------------------------

autoclass_content = "both"

autodoc_default_options = {
    "show-inheritance": True,
}

# -- Copybutton ----------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
