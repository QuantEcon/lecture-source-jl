# "Lectures in Quantitative Economics": Julia Version

## About this Repository 

This is the source repository for the QuantEcon Julia-flavored [lectures](https://lectures.quantecon.org/jl).

See `LICENSE.md` for licensing and copyright information. 

## Usage

### Prerequisities

* The latest `quantecon/jupinx` Docker image (see the **Containerization** section), or: 

     * Julia 1.2.x

     * Jupyter and Jupyter Lab

     * Jupinx (`sphinxcontrib-jupyter` on PyPi) 0.4.2 or later. 

     * The `make` command-line utility.

It's recommended that you install and precompile the packages used by the lectures **before** building. To do this: 

1. (Optional) Delete your `~/.julia` folder to start fresh.

2. `cd` to the `source/rst` folder in this repo. In a Julia REPL, run `] add InstantiateFromURL IJulia` and `] precompile`. 

3. Then, run `] activate .`, followed by `] instantiate` and `] precompile`. 
 
### Building

There are a few different targets, notably: 

* `make website`, which will generate Jupyter, execute notebooks, and then turn that into HTML 

* `make coverage`, which will do steps (1) and (2) above (**with otherwise hidden unit tests**), and then generate a report about which notebooks fail. 

* `make preview`, which will do steps (1), (2), and (3) above and then fire up a local HTTP server. 

### Options and Special Cases

Specifying parallel execution (i.e., `make coverage -e parallel=True`) will use 4 cores instead of 1. This leads to a notable speedup in build times. 

You can build only a few notebooks by [**FILL IN PROCEDURE HERE**]

### Containerized Build

Alternately, you can use the `quantecon/jupinx` docker image, which has all these dependencies baked in. 

The advantage of a containerized setup is that you can use a siloed, "pre-baked" setup environment to build the lectures. 

0. Install [Docker](https://www.docker.com/).

1. Run `docker pull quantecon/jupinx`. 

2. In a terminal, cd to this repository, and run `docker run --name quantecon-docker -it -d -v "$(pwd)":/home/jovyan/work quantecon/jupinx` from inside the directory (Linux/macOS). It should spit out a container ID string then exit. Try `${PWD}` on Windows, but your mileage may vary. 

     :warning: In order to guarantee reproducibility, you should either be mounting a fresh clone of this repository, or sanitize things by running `git clean -xdff` (remove uncommitted/untracked files) and `git reset --hard` (reset to the last git state.) Otherwise, local variance in the mounted files may impact your results.

3. In the same terminal (i.e., not inside the container), run `docker exec quantecon-docker bash -c "cd work && make jupyter".` Change it to `jupyter-tests` if you want it to output/execute the test blocks. 

4. Grab a coffee. The Julia side executes in serial, so it takes about an hour (modulo your processor speed.)

5. After it's done, in a terminal run `docker stop quantecon-docker` and `docker rm quantecon-docker`. This will garbage-collect the container, and free the name `quantecon-docker` for your next run. If you're having trouble, run `docker rm -f quantecon-docker` to force removal. 

