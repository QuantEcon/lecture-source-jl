# "Quantitative Economics with Julia":

## About this Repository 

This is the source repository for [Quantitative Economics with Julia](https://julia.quantecon.org).  These instructions required for authoring/editing the textbook and notebooks, and are not necessary for typical usage.

See `LICENSE.md` for licensing and copyright information.

## Release

For information on releasing a new lecture version, see [the docs](RELEASE.md).

## Usage

### WSL if on Windows
If on Windows, use WSL 2.

To get "Ubuntu on Windows" and other linux kernels see [instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10) and then https://docs.microsoft.com/en-us/windows/wsl/wsl2-install to install WSL2.

Hint on copy-paste:  One way to paste into a Windows terminal (of any sort) is the `<ctrl-c>` text somewhere else and then, while selected in the terminal at the cursor, to `<right click>` the mouse (which pastes)

When running the ubuntu shell run it in `Powershell` as an administrator

### Prerequisities

1. Start within your home directory (using WSL if on Windows, make sure to  `run as administrator`).

2. Go to your home directory and make sure key dependencies are installed
```bash
cd
sudo apt update
sudo apt install make gcc unzip
sudo apt-get update
sudo apt-get install libxt6 libxrender1 libgl1-mesa-glx libqt5widgets5 
```

2. Install Conda

   -  In the Ubuntu terminal, first install python/etc. tools
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
   bash Anaconda3-2020.02-Linux-x86_64.sh
   ```
   -  Create a directory `.conda` by running `mkdir ~/.conda` if the warning "Unable to register the environment" shows up
3. The installation will take time. You should:
   - accept default paths
   - accept licensing terms
   - *IMPORTANT* Manually choose `yes` to have it do the `conda init`
   - Delete the installation file
     ```bash
     rm Anaconda3-2020.02-Linux-x86_64.sh
     ```

4. Install Julia
```bash
wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.1-linux-x86_64.tar.gz | tar -xzv
```

4. Assuming you installed anaconda in your home directory then,
- Within your home directory, `edit .bashrc`.  This opens Vim.  Go to the bottom of the file, and type `i` to enter insert mode.
- Add something like the following:

```bash
export PATH=~/anaconda3/bin:~/julia-1.4.1/bin:$PATH
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

Precompile the packages used by the lectures **before** building. To do this: 

1. (Optional) Delete your `~/.julia` folder to start fresh.

2. In a Julia REPL (i.e. `julia` in terminal if your `.bashrc` was edited above), run

```julia
] add InstantiateFromURL IJulia; precompile
```

3. Start a new REPL

(If you didn't run the package compilation step, then `cd lecture-source-jl/source/rst`)
In the REPL, run

```julia
] activate .; instantiate; precompile
```
This will take a long time to run.  You can safely ignore build errors for `Electron`
 
**You may see a lot of warnings** during this step if you chose to use PackageCompiler acceleration above. They can be safely ignored.
 

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
- Consider adding the [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension for viewing the html from `_build/website/jupyter_html` files

### Options and Special Cases

Specifying parallel execution (i.e., `make coverage parallel=8`) will use 8 cores instead of 1. This leads to a notable speedup in build times. (There are some [`zmq` errors](https://github.com/QuantEcon/sphinxcontrib-jupyter/issues/261) that sporadically pop up at very high core counts, i.e. above 8.)

You can build only a few notebooks by `jupinx -w --files source/rst/<file>.rst`.
