# "Lectures in Quantitative Economics"-- Julia Version

## About this Repository 

This is the source repository for the QuantEcon Julia-flavored macroeconomics lectures. 

See `LICENSE.md` for licensing and copyright information. 

## Usage

0. Follow the [local setup instructions](https://lectures.quantecon.org/jl/getting_started.html) to get setup with Anaconda Python and Julia **1.1**. Install [Jupinx](https://github.com/QuantEcon/sphinxcontrib-jupyter).

1. Clone or download the repo and run `make setup` to instantiate necessary links, etc.

2. Run `make jupyter` to generate notebooks without tests, and `make jupyter-tests` to generate notebooks with tests.

## Contributing

All contributions should be made through PR. If you aren't an admin on the repo, you might need an admin review before the PR is mergeable. You can request reviews from the right hand dropdown.

Make sure you have a look at the [style guide](style.md) before you start writing. The [unicode](unicode.jl) is a concise summary of which unicode patterns we use.
