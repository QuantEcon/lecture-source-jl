# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = lecture-source-jl
SOURCEDIR     = rst_files
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

jupyter:
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

jupyter-tests:
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D jupyter_drop_tests=0

preview:
	cd _build/jupyter_html/ && python -m http.server

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

