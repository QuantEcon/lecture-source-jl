#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# QuantEcon.lectures-python documentation build configuration file, created by
# sphinx-quickstart on Mon Feb 13 14:28:35 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import nbformat
import datetime

now = datetime.datetime.now()


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.5'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.mathjax',
	'sphinxcontrib.bibtex',
	'IPython.sphinxext.ipython_console_highlighting',
    # Custom Sphinx Extensions
    'sphinxcontrib.jupyter',
]

# Retired Extensions but may be useful in Future

	# 'matplotlib.sphinxext.plot_directive',
	# 'matplotlib.sphinxext.only_directives',
	# 'sphinxcontrib.tikz',
	# 'sphinx.ext.graphviz',

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'
master_pdf_doc = 'index'

# General information about the project.
project = 'QuantEcon.lectures-julia'
copyright = '2017, Thomas J. Sargent and John Stachurski'
author = 'Thomas J. Sargent and John Stachurski'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '%s-%s-%s' % (now.year, now.strftime("%b"), now.day)
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '_static']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Add rst prolog
rst_prolog = """
.. highlight:: julia
"""

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'qe-lectures'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['_themes']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Quantitative Economics"

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {
#         'index': ['jl_layout.html'],
#     }

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'QuantEconlectures-juliadoc'

# Tikz HTML configuration for rendering images
tikz_latex_preamble = r"""
    \usetikzlibrary{arrows}
    \usetikzlibrary{calc}
    \usetikzlibrary{intersections}
    \usetikzlibrary{decorations}
    \usetikzlibrary{decorations.pathreplacing}
"""

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
'pointsize': '11pt',

# Additional stuff for the LaTeX preamble.
'preamble': r"""
\usepackage{amsmath, amssymb}
\usepackage{mathrsfs}

\usepackage{tikz}
\usetikzlibrary{arrows}
\usetikzlibrary{calc}
\usetikzlibrary{intersections}
\usetikzlibrary{decorations}
\usepackage{pgf}
\usepackage{pgfplots}
\usepackage{bbm}

\newcommand{\NN}{\mathbbm N}
\newcommand{\PP}{\mathbbm P}
\newcommand{\EE}{\mathbbm E \,}
\newcommand{\XX}{\mathbbm X}
\newcommand{\ZZ}{\mathbbm Z}
\newcommand{\QQ}{\mathbbm Q}

\newcommand{\fF}{\mathcal F}
\newcommand{\dD}{\mathcal D}
\newcommand{\lL}{\mathcal L}
\newcommand{\gG}{\mathcal G}
\newcommand{\hH}{\mathcal H}
\newcommand{\nN}{\mathcal N}
\newcommand{\pP}{\mathcal P}

\DeclareMathOperator{\trace}{trace}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Span}{span}
\DeclareMathOperator{\proj}{proj}
\DeclareMathOperator{\col}{col}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

\usepackage{makeidx}
\makeindex
""",

# Latex figure (float) alignment (Could use 'H' to force the placement of figures)
'figure_align': 'H',#'htbp',

#Add Frontmatter before TOC
'tableofcontents' : r"""\newpage
\thispagestyle{empty}
\chapter*{Preface}
\large
This \textbf{pdf} presents a series of lectures on quantitative economic
modeling, designed and written by \href{http://www.tomsargent.com/}{Thomas J. Sargent} and \href{http://johnstachurski.net}{John Stachurski}.
The primary programming languages are \href{https://www.python.org}{Python} and \href{http://julialang.org/}{Julia}.
You can send feedback to the authors via contact@quantecon.org.

\vspace{5em}

\begin{leftbar}
\textbf{Note: You are currently viewing an automatically generated
pdf version of our online lectures,} which are located at

\vspace{2em}

\begin{center}
  \texttt{https://lectures.quantecon.org}
\end{center}

\vspace{2em}

Please visit the website for more information on the aims and scope of the
lectures and the two language options (Julia or Python).

\vspace{1em}

Due to automatic generation of this pdf, \textbf{presentation quality is likely
to be lower than that of the website}.

\end{leftbar}

\normalsize

\sphinxtableofcontents
"""
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_pdf_doc, 'QuantEconlectures-julia.tex', 'QuantEcon.lectures-julia PDF',
     'Thomas J. Sargent and John Stachurski', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True

# ------------------
# Linkcheck Options
# ------------------

linkcheck_ignore = [r'https:\/\/github\.com\/.*?#.*'] #Anchors on Github seem to create issues with linkchecker

linkcheck_timeout = 30

# --------------------------------------------
# jupyter Sphinx Extension conversion settings
# --------------------------------------------

# Conversion Mode Settings
# If "all", convert codes and texts into notebook
# If "code", convert codes only
jupyter_conversion_mode = "all"

jupyter_write_metadata = False

# Location for _static folder
jupyter_static_file_path = ["source/_static", "_static"]

# default lang
jupyter_default_lang = "julia"

# Configure Jupyter Kernels
jupyter_default_lang = "julia"

jupyter_kernels = {
    "julia": {
        "kernelspec": {
            "display_name": "Julia 1.2.0",
            "language": "julia",
            "name": "julia-1.2"
            },
        "file_extension": ".jl"
    }
}

# Configure jupyter headers
jupyter_headers = {
    "python3": [
        # nbformat.v4.new_code_cell("%autosave 0")      #@mmcky please make this an option
        ],
    "julia": [
        ],
}

# Filename for the file containing the welcome block
jupyter_welcome_block = ""

#Adjust links to target html (rather than ipynb)
jupyter_target_html = False

#path to download notebooks from 
jupyter_download_nb_urlpath = None

#allow downloading of notebooks
jupyter_download_nb = False

#Use urlprefix images
jupyter_download_nb_image_urlpath = None

#Allow ipython as a language synonym for blocks to be ipython highlighted
jupyter_lang_synonyms = ["ipython"]

#Execute skip-test code blocks for rendering of website (this will need to be ignored in coverage testing)
jupyter_ignore_skip_test = True

#allow execution of notebooks
jupyter_execute_notebooks = False

# Location of template folder for coverage reports
jupyter_template_coverage_file_path = False

# generate html from IPYNB files
jupyter_generate_html = False

# html template specific to your website needs
jupyter_html_template = None

# latex template specific to your website needs
jupyter_latex_template = ""

#make website
jupyter_make_site = False

#force markdown image inclusion
jupyter_images_markdown = True

#This is set true by default to pass html to the notebooks
jupyter_allow_html_only=True

## Theme specific variables
jupyter_theme_path = 'theme'
jupyter_template_path = 'theme/templates'

# Jupyter dependencies
jupyter_dependencies = {
    '': ['Manifest.toml', 'Project.toml'],
    'dynamic_programming': [],
    'dynamic_programming_squared': [],
    'getting_started_julia': [],
    'more_julia': [],
    'multi_agent_models': [],
    'time_series_models': [],
    'tools_and_techniques': []
}

jupyter_download_nb_execute=True

# PDF options

jupyter_pdf_logo = "_static/img/qe-menubar-logo.png"

jupyter_bib_file = "_static/quant-econ"

jupyter_pdf_author = "Thomas J. Sargent and John Stachurski"