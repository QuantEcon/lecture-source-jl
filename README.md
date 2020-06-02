# "Quantitative Economics with Julia":

## About this Repository 

This is the source repository for [Quantitative Economics with Julia](https://julia.quantecon.org).  These instructions required for authoring/editing the textbook and notebooks, and are not necessary for typical usage.

See `LICENSE.md` for licensing and copyright information.

## Release

For information on releasing a new lecture version, see [the docs](RELEASE.md).

## Usage

### WSL and VSCode if on Windows
- To get "Ubuntu on Windows" and other linux kernels see [instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
  - For the installation, run it in `Powershell` as an administrator
  - Hint on copy-paste:  One way to paste into a a windows (of any sort) is the `<ctrl-c>` text somewhere else and then, while selected in the terminal at the cursor, to `<right click>` the mouse (which pastes).
- Install [VSCode](https://code.visualstudio.com/) with remote WSL support on windows
 - See [VS Code Remote Editing](https://code.visualstudio.com/docs/remote/remote-overview)
 - [VS Code Remote WSL](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl)


To open the WSL in VS Code
- Click on the "><" icon on the bottom left hand corner, and open the remote folder in your WSL image (e.g. `~/lecture-source-jl`)
- Choose "TERMINAL" to open a [WSL terminal](https://code.visualstudio.com/docs/remote/wsl#_opening-a-terminal-in-wsl), and run any of the above jupinx or make commands.

To get git credentials integrated, in a windows terminal (i.e. not in WSL) run
```
 git config --global credential.helper wincred
```
Then in a WSL terminal (within VS Code or otherwise),
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-wincred.exe"
```
(See more details in [Sharing Credentials](https://code.visualstudio.com/docs/remote/troubleshooting#_sharing-git-credentials-between-windows-and-wsl) )

### Prerequisities

1. Start within your home directory in linux or in a WSL terminal

2. Go to your home directory and make sure key dependencies are installed
```bash
cd ~
sudo apt update
sudo apt-get upgrade
sudo apt install make gcc unzip
sudo apt-get update
sudo apt-get install libxt6 libxrender1 libgl1-mesa-glx libqt5widgets5 
```sudo 

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
     sudo pip install --upgrade --force-reinstall pyzmq
     rm Anaconda3-2020.02-Linux-x86_64.sh
     ```

4. Install Julia
```bash
wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.2-linux-x86_64.tar.gz | tar -xzv
```

4. Assuming you installed anaconda in your home directory then,
- Within your home directory modify the `.bashrc` (with VS Code, you can edit directly after connecting to WSL)
- Add something like the following:

```bash
export PATH=~/anaconda3/bin:~/julia-1.4.2/bin:$PATH
```

Then, from your terminal, run `source .bashrc` to load the changes in the current WSL terminal.

5. Install Jupinx and deps

```bash
conda upgrade conda
conda install dask distributed
pip install jupinx sphinxcontrib.bibtex jupinx guzzle_sphinx_theme

```

6. Clone the repo to your preferred location (note that WSL+vscode+ssh cloning has bugs, so use https).

On Windows, you want to clone to the WSL setup, not onto your windows drive.

```bash
git clone https://github.com/QuantEcon/lecture-source-jl
```

Precompile the packages used by the lectures **before** building. To do this: 

1. (Optional) If you have a major preexisting julia setup, consider deleting `~/.julia` folder to start fresh.

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

### Building

There are a few different targets, notably: 

* `make website`, which will generate Jupyter, execute notebooks, and then turn that into HTML 

* `make coverage`, which will do steps (1) and (2) above (**with otherwise hidden unit tests**), and then generate a report about which notebooks fail. 

* `make preview`, which will do steps (1), (2), and (3) above and then fire up a local HTTP server. 

* `jupinx -w --files source/rst/getting_started_julia/julia_by_example.rst`, or any other `.rst` for a single file

### Workflow for Editing with VS Code
*Setup*
A few useful extensions to install (in WSL if on windows):
- https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer
- See https://github.com/ubcecon/tutorials/blob/master/vscode.md for general extensions as well as Julia and Python
- https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext which requires the python extension
  - Note that for the RST extension you will need to have python extension installed.  When it asks for the python interpreter, choose `DocUtils` rather than sphinx

*One Workflow*:
- When you open the project, in the julia terminal go `;cd source/rst` and then `] activate .` to ensure you are using the package/manifest.
- Edit for the content.  With the RST extension, you can get a preview (`Ctrl+K Ctrl+R`, or clicking on the preview buton)
  - It will have plenty of errors since it doesn't apply jupinx, but will help avoid obvious RST issues.
- When ready to edit the code, use `Ctrl+Shift+P` and type `Change Language Mode` to change it to Julia
  - With this, you can use `Shift+Enter,Ctrl+Enter`or `Alt+Enter` depending on your setup, to run one line of the code at a time in the Julia REPL.  If you activated the Package in above, it will use the correct versions of those packages.
- When you want to see the results, run in the bash script to build the files
```bash
jupinx -w --files source/rst/getting_started_julia/fundamental_types.rst
```
- And finally, you can watch the file with the LiveServer extension.  Go to the `_build/webiste/jupyter_html/getting_started_julia/fundamental_types.htm` etc. in the file explorer, and right click to "Open with LiveServer"

### Options and Special Cases

Specifying parallel execution (i.e., `make coverage parallel=8`) will use 8 cores instead of 1. This leads to a notable speedup in build times. (There are some [`zmq` errors](https://github.com/QuantEcon/sphinxcontrib-jupyter/issues/261) that sporadically pop up at very high core counts, i.e. above 8.)

You can build only a few notebooks by `jupinx -w --files source/rst/<file>.rst`.
