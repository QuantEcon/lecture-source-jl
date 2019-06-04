# "Lectures in Quantitative Economics": Julia Version

## About this Repository 

This is the source repository for the QuantEcon Julia-flavored macroeconomics lectures. 

See `LICENSE.md` for licensing and copyright information. 

## Usage

0. Install [Docker](https://www.docker.com/).

1. Run `docker pull quantecon/jupinx`. 

2. In a terminal, run `docker run --name quantecon-docker -it -d -v "$(pwd)":/home/jovyan/work quantecon/jupinx` from inside the directory (Linux/macOS). It should spit out a container ID string then exit. Try `${PWD}` on Windows, but note that **Windows isn't mentioned in the [docs](https://docs.docker.com/storage/bind-mounts/) for bind mounts.**  

3. In the same terminal (i.e., not inside the container), run `docker exec quantecon-docker bash -c "cd work && make jupyter".` Change it to `jupyter-tests` if you want it to output/execute the test blocks. 

4. After it's done, in the same terminal run `docker stop quantecon-docker` and `docker rm quantecon-docker`. This will garbage-collect the container, and free the name `quantecon-docker` for your next run. If you're having trouble, run `docker rm -f quantecon-docker` to force removal. 

