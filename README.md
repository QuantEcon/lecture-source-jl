# "Lectures in Quantitative Economics"-- Julia Version


## About this Repository 

This is the source repository for the QuantEcon datascience lectures. 

## Usage

To build the lectures from source: 

0. Open a terminal (powershell on windows) and `cd` to the location of this file

1. `docker pull arnavsood/jupinx:latest`

2. Start the container:
  - Powershell: `docker run --name quantecon-docker -it -d -v ${PWD}:/home/jovyan/work arnavsood/jupinx`
  - Linux/Mac: `docker run --name quantecon-docker -it -d -v "$(pwd)":/home/jovyan/work arnavsood/jupinx`

3. Then `docker exec quantecon-docker bash -c "cd work && make jupyter"` (`jupyter-tests` for tests).

4. Once you're done, `docker rm -f quantecon-docker`.

The above includes a call to `make jupyter`.

## Development Tools

We provide `run-all-notebooks.sh`, `run-all-inplace.sh`, and `run-notebook.sh` (which you can run from within the docker by adapting step (3) above, or by starting a terminal with `docker exec -it quantecon-docker /bin/bash` (`/bin/sh` on Powershell?). 

## Contributing

All contributions should be made through PR. If you aren't an admin on the repo, you might need an admin review before the PR is mergeable. You can request reviews from the right hand dropdown.
