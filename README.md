# "Lectures in Quantitative Economics"-- Julia Version

## Setup Instructions
1. Clone this repo
2. Install Conda and Julia on your target operating system.  If on Windows, install the [Windows Substyem for Linux](https://github.com/econtoolkit/tutorials/blob/master/julia/WSL.md)
3. Assuming that python is in your path, install the dependencies `pip install sphinxcontrib-bibtex`
4. Clone with `git clone https://github.com/QuantEcon/sphinxcontrib-jupyter`
5. `cd` to the `sphinxcontrib-jupyter`
6. Install with `python setup.py install`
7. Go to the `notebooks` directory of this repo, run `julia` and then install either `include("instantiate.jl")` or run
```julia
using Pkg
pkg"add IJulia; precompile"
pkg"activate .;instantiate; precompile"
```
This may take a long time, as it will install and precompile every package used by the lecture notes.

## Generating, testing, and executing the notebooks
- To generate the notebooks and execute them
  1. In the main directory of the repo: `make jupyter`
  2. Execute all of the notebooks with `./run-all-notebooks.sh`.  This will take a long-time the first execution.
  3. `cd` to the `/notebooks` directory, and run `jupyter lab` to examine them. 
- To generate a "production" version of the notebooks without the tests
  1. Edit `rst_files/conf.py` and modify the setting `jupyter_drop_tests = True`
  2. In the main directory of the repo: `make jupyter``
  3. Execute all of the notebooks with `./run-all-notebooks.sh`.  This will take a long-time the first execution.
  4. `cd` to the `/notebooks` directory, and run `jupyter lab` to examine them. 

## Development Tools

On top of the above, we have the following dev scripts: 