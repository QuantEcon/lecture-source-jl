# "Lectures in Quantitative Economics": Julia Version

## About this Repository 

This is the source repository for the QuantEcon Julia-flavored [lectures](https://lectures.quantecon.org/jl).

See `LICENSE.md` for licensing and copyright information. 

## Usage

0. Install [Docker](https://www.docker.com/).

1. Run `docker pull quantecon/jupinx`. 

2. In a terminal, cd to this repository, and run `docker run --name quantecon-docker -it -d -v "$(pwd)":/home/jovyan/work quantecon/jupinx` from inside the directory (Linux/macOS). It should spit out a container ID string then exit. Try `${PWD}` on Windows, but your mileage may vary. 

     :warning: In order to guarantee reproducibility, you should either be mounting a fresh clone of this repository, or sanitize things by running `git clean -xdff` (remove uncommitted/untracked files) and `git reset --hard` (reset to the last git state.) Otherwise, local variance in the mounted files may impact your results.

3. In the same terminal (i.e., not inside the container), run `docker exec quantecon-docker bash -c "cd work && make jupyter".` Change it to `jupyter-tests` if you want it to output/execute the test blocks. 

4. Grab a coffee. The Julia side executes in serial, so it takes about an hour (modulo your processor speed.)

5. After it's done, in a terminal run `docker stop quantecon-docker` and `docker rm quantecon-docker`. This will garbage-collect the container, and free the name `quantecon-docker` for your next run. If you're having trouble, run `docker rm -f quantecon-docker` to force removal. 

