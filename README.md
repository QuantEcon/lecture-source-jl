# "Quantitative Economics with Julia":

## About this Repository 

This is the source repository for [Quantitative Economics with Julia](https://lectures.quantecon.org/jl).  These instructions required for authorig/editing the textbook and notebooks, and are not necessary for typical usage.

See `LICENSE.md` for licensing and copyright information. 

## Usage

### Prerequisities

* The latest `quantecon/jupinx` Docker image (see the **Containerization** section), or: 

0. Start within your home directory, using [WSL](https://github.com/ubcecon/cluster_tools/blob/master/WSL.md#install-wsl-from-ubuntu-and-conda) if on Windows. If you're running from the shell, make sure you `run as administrator`.

1. Go to your home directory and make sure key dependencies are installed
```bash
cd
sudo apt install make 
sudo apt-get update
sudo apt-get install libxt6 libxrender1 libgl1-mesa-glx libqt5widgets5 
```

2. Install Conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh
```
Choose `yes` to: "Do you wish the installer to initialize Anaconda3 by running conda init?"

3. Install Julia
```bash
wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz | tar -xzv
```

4. Assuming you installed anaconda in your home directory then,
- Within your home directory, `edit .bashrc`.  This opens Vim.  Go to the bottom of the file, and type `i` to enter insert mode.
- Add something like the following:

```bash
export PATH=~/anaconda3/bin:~/julia-1.2.0/bin:$PATH
```
Hit `<Esc>` to exit insert mode, and then type `:x` to save and exit.

Then, from your terminal, run `source .bashrc` to load the changes in the current WSL terminal.

5. Install Jupinx and deps
```bash
conda upgrade conda
pip install jupinx
pip install sphinxcontrib.bibtex
conda install dask distributed
```

6. Clone the repo to your preferred location (note that WSL+vscode+ssh cloning has bugs, so use https)

```bash
git clone https://github.com/QuantEcon/lecture-source-jl
```


It's recommended that you install and precompile the packages used by the lectures **before** building. To do this: 

1. (Optional) Delete your `~/.julia` folder to start fresh.

2. `cd` to the `source/rst` folder in this repo. In a Julia REPL (i.e. `julia` in terminal if your `.bashrc` was edited above), run

```julia
] add InstantiateFromURL IJulia; precompile
```

3. Then (verifying you are in the `/lecture-source-jl/source/rst` diirectory), still inside the Julia REPL, run

```julia
] activate .; instantiate; precompile
```
This will take a long time to run.  You can safely ignore build errors for `Electron`
 
### Building

There are a few different targets, notably: 

* `make website`, which will generate Jupyter, execute notebooks, and then turn that into HTML 

* `make coverage`, which will do steps (1) and (2) above (**with otherwise hidden unit tests**), and then generate a report about which notebooks fail. 

* `make preview`, which will do steps (1), (2), and (3) above and then fire up a local HTTP server. 

* `jupinx -w --files source/rst/getting_started_julia/julia_by_example.rst`, or any other `.rst` for a single file

### Editing with WSL and VS Code
See [VS Code Remote Editing](https://code.visualstudio.com/docs/remote/remote-overview) and [VS Code Remote WSL](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl)

In a windows terminal run
```
 git config --global credential.helper wincred
```
In a WSL terminal,
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-wincred.exe"
```
(see more details in [Sharing Credentials](https://code.visualstudio.com/docs/remote/troubleshooting#_sharing-git-credentials-between-windows-and-wsl) )

To open the WSL in VS Code
- Click on the "><" icon on the bottom left hand corner, and open the remote folder in your WSL image (e.g. `~/lecture-source-jl`)
- Choose "TERMINAL" to open a [WSL terminal](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl), and run any of the above jupinx or make commands.
- Consider adding a [RST Extension](https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext)

### Options and Special Cases

Specifying parallel execution (i.e., `make coverage parallel=8`) will use 8 cores instead of 1. This leads to a notable speedup in build times. (There are some [`zmq` errors](https://github.com/QuantEcon/sphinxcontrib-jupyter/issues/261) that sporadically pop up at very high core counts, i.e. above 8.)

You can build only a few notebooks by `jupinx -w --files source/rst/<file>.rst`.

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

