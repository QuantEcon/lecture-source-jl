# "Lectures in Quantitative Economics"-- Julia Version

## Setup Instructions
1. Clone this repo
2. Install Conda and Julia on your target operating system.  If on Windows, install the [Windows Substyem for Linux](https://github.com/econtoolkit/tutorials/blob/master/julia/WSL.md)
3. Assuming that python is in your path, install the dependencies `pip install sphinxcontrib-bibtex`
4. Clone with `git clone https://github.com/QuantEcon/sphinxcontrib-jupyter`
5. `cd` to the `sphinxcontrib-jupyter`
6. Install with `python setup.py install`
7. In a Julia terminal run

```julia
using Pkg
pkg"add IJulia; precompile"
pkg"add InstantiateFromURL; precompile"
using InstantiateFromURL
activate_github("QuantEcon/QuantEconLecturePackages", tag = "v0.9.7")
activate_github("QuantEcon/QuantEconLectureAllPackages", tag = "v0.9.7")
```

This may take a long time, as it will install and precompile every package used by the lecture notes.

## Generating, testing, and executing the notebooks
- To generate the notebooks and execute them
  1. In the main directory of the repo: `make jupyter` or `make jupyter-tests` if you want it with testing code
  2. Execute all of the notebooks with `./run-all-notebooks.sh`.  This will take a long-time the first execution.
  3. `cd` to the `/notebooks` directory, and run `jupyter lab` to examine them.

## Contributing

All contributions should be made through PR. If you aren't an admin on the repo, you might need an admin review before the PR is mergeable. You can request reviews from the right hand dropdown.
