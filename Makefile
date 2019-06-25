# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = lecture-source-jl
SOURCEDIR     = source/rst
BUILDDIR      = _build
CORES 		  = 4
BUILDCOVERAGE = _build_coverage


# Put it first so that "make" without argument is like "make help".
setup:
	cd source/rst && ln -s ../_static _static

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

jupyter:
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

jupyter-tests:
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D jupyter_drop_tests=0

preview:
	cd _build/jupyter_html/ && python -m http.server

jupyter-parallel:
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D jupyter_number_workers=$(CORES)

clean-coverage:
	rm -rf $(BUILDCOVERAGE)

coverage: clean-coverage
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDCOVERAGE)" $(SPHINXOPTS) $(O) -D jupyter_make_coverage=1 -D jupyter_make_site=0 -D jupyter_generate_html=0 -D jupyter_ignore_skip_test=0 -D jupyter_drop_tests=0 -D jupyter_download_nb=0


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

