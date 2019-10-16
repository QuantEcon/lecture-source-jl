.. _tools_editors:

.. include:: /_static/includes/header.raw

******************************************
Julia Tools and Editors
******************************************

.. contents:: :depth: 2

Co-authored with Arnav Sood

While Jupyter notebooks are a great way to get started with the language, eventually you will want to use more powerful tools.

We'll discuss a few of them here, such as

* Text editors like Atom, which come with rich Julia support for debugging, documentation, git integration, plotting and inspecting data, and code execution.

* The Julia REPL, which has specialized modes for package management, shell commands, and help.

.. * A virtualized Docker setup which provides a painless pre-configured environment on your machine.

Note that we assume you've already completed the :doc:`getting started <../getting_started_julia/getting_started>` and :doc:`interacting with Julia <../getting_started_julia/julia_environment>` lectures.

Preliminary Setup
====================

Follow the instructions for setting up Julia :ref:`on your local computer <jl_jupyterlocal>`.

.. _jl_startup_file:

Creating a Startup File (Recommended)
----------------------------------------------------

Whenever the Julia compiler or REPL starts, it will look for a file called ``startup.jl`` (see `Julia Manual <https://docs.julialang.org/en/v1/manual/getting-started/#man-getting-started-1>`_).

We provide a file here which does two things

* Makes the REPL shell mode "sticky," so you don't need to keep running ``;`` for new commands.

* Loads the ``Revise.jl`` package on startup, which lets you see changes you make to a package in real-time (i.e., no need to quit the REPL, open again, and load again).

The location for the file is relative to your default Julia environment (e.g. ``~/.julia/config/startup.jl`` or ``C:\Users\USERNAME\.julia\config\startup.jl`` on Windows).

Recall that you can find the location of the ``~/.julia`` directory by running

.. code-block:: julia

    DEPOT_PATH[1]

**Note:** On Mac, this won't be visible in the Finder unless you specifically enable that option, but you can get to it by running ``cd .julia; open .`` from a new terminal.

To add the file:

* In the ``julia`` terminal, type the following

    .. code-block:: none

        ] add  Revise REPL; precompile

* Create the ``~/.julia/config/`` directory if necessary in the terminal or file explorer.

* Download the file `startup.jl <https://s3-ap-southeast-2.amazonaws.com/lectures.quantecon.org/jl/_static/includes/startup.jl>`_ into that directory.

* For convenience, you may find it useful on your operating system to change the directory where the REPL starts.

On Windows, if you have a shortcut on your desktop or on the taskbar, you could: (1) right-click on the icon; (2) right click on the "julia" text; (3) choose "Properties", and (4) change the "Start In" to be something such as ``C:\Users\YOURUSERNAME\Documents``.

.. _repl_main:

The REPL
=============

Previously, we discussed basic use of the Julia REPL ("Read-Evaluate-Print Loop").

Here, we'll consider some more advanced features.

Shell Mode
-----------------

Hitting ``;`` brings you into shell mode, which lets you run bash commands (PowerShell on Windows)

.. code-block:: julia

    ; pwd

You can also use Julia variables from shell mode

.. code-block:: julia

    x = 2

.. code-block:: julia

    ; echo $x

Package Mode
-----------------

Hitting ``]`` brings you into package mode.

* ``] add Expectations`` will add a package (here, ``Expectations.jl``).

* Likewise, ``] rm Expectations`` will remove that package.

* ``] st`` will show you a snapshot of what you have installed.

* ``] up`` will (intelligently) upgrade versions of your packages.

* ``] precompile`` will precompile everything possible.

* ``] build`` will execute build scripts for all packages.

* Running ``] preview`` before a command (i.e., ``] preview up``) will display the changes without executing.

You can get a full list of package mode commands by running

.. code-block:: julia

    ] ?

On some operating systems (such as OSX) REPL pasting may not work for package mode, and you will need to access it in the standard way (i.e., hit ``]`` first and then run your commands).

Help Mode
---------------

Hitting ``?`` will bring you into help mode.

The key use case is to find docstrings for functions and macros, e.g.

.. code-block:: julia
    :class: no-execute

    ? print

Note that objects must be loaded for Julia to return their documentation, e.g.

.. code-block:: julia
    :class: no-execute

    ? @test

will fail, but

.. code-block:: julia
    :class: no-execute

    using Test

.. code-block:: julia
    :class: no-execute

    ? @test

will succeed.

Atom
=========

As discussed :doc:`previously <../getting_started_julia/getting_started>`, eventually you will want to use a fully fledged text editor.

The most feature-rich one for Julia development is `Atom <https://atom.io/>`_, with the `Juno <http://junolab.org/>`_ package.

There are several reasons to use a text editor like Atom, including

* Git integration (more on this in the :doc:`next lecture <../more_julia/version_control>`).

* Painless inspection of variables and data.

* Easily run code blocks, and drop in custom snippets of code.

* Integration with Julia documentation and plots.

Installation and Configuration
---------------------------------

Installing Atom
^^^^^^^^^^^^^^^^^^^

1. Download and Install Atom from the `Atom website <https://atom.io/>`_.

2. (Optional, but recommended): Change default Atom settings

    * Use ``Ctrl-,`` to get the ``Settings`` pane
    * Choose the ``Packages`` tab
    * Type ``line-ending-selector`` into the Filter and then click "Settings" for that package

        * Change the default line ending to ``LF`` (only necessary on Windows)

    * Choose the `Editor` tab

        * Turn on ``Soft Wrap``
        * Set the ``Tab Length`` default to ``4``

Installing Juno
^^^^^^^^^^^^^^^^^^^^^^^

1. Use ``Ctrl-,`` to get the `Settings` pane.
2. Go to the ``Install`` tab.
3. Type ``uber-juno`` into the search box and then click `Install` on the package that appears.
4. Wait while Juno installs dependencies.
5. When it asks you whether or not to use the standard layout, click ``yes``.

At that point, you should see a built-in REPL at the bottom of the screen and be able to start using Julia and Atom.

Troubleshooting
^^^^^^^^^^^^^^^^^^^
Sometimes, Juno will fail to find the Julia executable (say, if it's installed somewhere nonstandard, or you have multiple).

To do this
1. ``Ctrl-,`` to get `Settings` pane, and select the `Packages` tab.
2. Type in ``julia-client`` and choose `Settings`.
3. Find the `Julia Path`, and fill it in with the location of the Julia binary.

    * To find the binary, you could run ``Sys.BINDIR`` in the REPL, then add in an additional ``/julia`` to the end of the screen.
    * e.g. ``C:\Users\YOURUSERNAME\AppData\Local\Julia-1.0.1\bin\julia.exe`` on Windows as ``/Applications/Julia-1.0.app/Contents/Resources/julia/bin/julia`` on OSX.

See the `setup instructions for Juno <http://docs.junolab.org/latest/man/installation.html>`_  if you have further issues.

If you upgrade Atom and it breaks Juno, run the following in a terminal. 

.. code-block:: none 

    apm uninstall ink julia-client
    apm install ink julia-client


Standard Layout
------------------

If you follow the instructions, you should see something like this when you open a new file.

If you don't, simply go to the command palette and type "Julia standard layout"

.. figure:: /_static/figures/juno-standard-layout.png
    :width: 100%

The bottom pane is a standard REPL, which supports the different modes above.

The "workspace" pane is a snapshot of currently-defined objects.

For example, if we define an object in the REPL

.. code-block:: julia

    x = 2

Our workspace should read

.. figure:: /_static/figures/juno-workspace-1.png
    :width: 100%

The ``ans`` variable simply captures the result of the last computation.

The ``Documentation`` pane simply lets us query Julia documentation

.. figure:: /_static/figures/juno-docs.png
    :width: 100%

The ``Plots`` pane captures Julia plots output (the code is as follows)

.. code-block:: julia
    :class: no-execute

    using Plots
    gr(fmt = :png);
    data = rand(10, 10)
    h = heatmap(data)

.. figure:: /_static/figures/juno-plots.png
    :width: 100%

**Note:** The plots feature is not perfectly reliable across all plotting backends, see `the Basic Usage <http://docs.junolab.org/latest/man/basic_usage.html>`_ page.

Other Features
-------------------

* `` Shift + Enter `` will evaluate a highlighted selection or line (as above).

* The run symbol in the left sidebar (or ``Ctrl+Shift+Enter``) will run the whole file.

See `basic usage <http://docs.junolab.org/latest/man/basic_usage.html>`_ for an exploration of features, and  the `FAQ <http://docs.junolab.org/latest/man/faq.html>`_ for more advanced steps.

.. Docker Integration
.. ----------------------
..
.. You can plug Juno/Atom into a Julia session running in a docker container, such as the QuantEcon base container.
..

.. _jl_packages:

Package Environments
========================

Julia's package manager lets you set up Python-style "virtualenvs," or subsets of packages that draw from an underlying pool of assets on the machine.

This way, you can work with (and specify) the dependencies (i.e., required packages) for one project without worrying about impacts on other projects.

* An ``environment`` is a set of packages specified by a ``Project.toml`` (and optionally, a ``Manifest.toml``).

* A ``registry`` is a git repository corresponding to a list of (typically) registered packages, from which Julia can pull (for more on git repositories, see :doc:`version control <../more_julia/version_control>`).

* A ``depot`` is a directory, like ``~/.julia``, which contains assets (compile caches, registries, package source directories, etc.).

Essentially, an environment is a dependency tree for a project, or a "frame of mind" for Julia's package manager.

* We can see the default (``v1.1``) environment as such

.. code-block:: julia

    ] st

* We can also create and activate a new environment

.. code-block:: julia

    ] generate ExampleEnvironment

* And go to it

.. code-block:: julia

    ; cd ExampleEnvironment

* To activate the directory, simply

.. code-block:: julia

    ] activate .

where "." stands in for the "present working directory".

* Let's make some changes to this

.. code-block:: julia

    ] add Expectations Parameters

Note the lack of commas

* To see the changes, simply open the ``ExampleEnvironment`` directory in an editor like Atom.

The Project TOML should look something like this

.. code-block:: none

    name = "ExampleEnvironment"
    uuid = "14d3e79e-e2e5-11e8-28b9-19823016c34c"
    authors = ["QuantEcon User <quanteconuser@gmail.com>"]
    version = "0.1.0"

    [deps]
    Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
    Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"

We can also

.. code-block:: julia

    ] precompile

**Note** The TOML files are independent of the actual assets (which live in ``~/.julia/packages``, ``~/.julia/dev``, and ``~/.julia/compiled``).

You can think of the TOML as specifying demands for resources, which are supplied by the ``~/.julia`` user depot.

* To return to the default Julia environment, simply

.. code-block:: julia

    ] activate

without any arguments.

* Lastly, let's clean up

.. code-block:: julia

    ; cd ..

.. code-block:: julia

    ; rm -rf ExampleEnvironment

InstantiateFromURL
-----------------------

With this knowledge, we can explain the operation of the setup block

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

What this ``github_project`` function does is activate (and if necessary, download, instantiate and precompile) a particular Julia environment.

.. _docker_main:

.. Docker
.. ===========

.. Docker is a tool that lets you run preconfigured, lightweight environments as applications on your computer or in a computational cloud.

.. The advantage of a Docker-based workflow is that it's perfectly reproducible, and that setup (of Julia versions and dependencies, etc.) is handled upstream by the image maintainer.

.. Here, we'll walk through the setup and installation steps, along with the main features of the ``quantecon/base`` Docker image.

.. Setup
.. -----------

.. * First, create an account for `Docker Hub <https://hub.docker.com/>`_ and create a docker id.

.. * Next, download and install Docker for

..     * `Mac <https://store.docker.com/editions/community/docker-ce-desktop-mac>`_.
..     * `Windows <https://store.docker.com/editions/community/docker-ce-desktop-windows>`_ - **do not** choose to use Windows containers.

.. **Note:** For Windows

..     * Hyper-V support should be enabled. For Windows 10 users, this means you must use a Pro, Enterprise, or Education version (**not Home or Mobile**).

..     * If you don't meet these requirements, the `Docker Toolbox for Windows <https://docs.docker.com/toolbox/toolbox_install_windows/>`_ may help.

.. * Next, to verify that there are no obvious errors in the installation, open a terminal (macOS/Linux) or Powershell (Windows) and run

..     .. code-block:: none

..         docker pull hello-world
..         docker run hello-world

.. You should see something like

.. .. figure:: /_static/figures/docker-hello-world.png
..     :width: 100%

.. * Then, download the QuantEcon Docker image by running the following in your terminal (this may take some time depending on your internet connection)

..     .. code-block:: none

..         docker pull quantecon/base

.. * Next, create a "data volume," or a hidden directory where Docker will persist any changes to your Julia packages

..     .. code-block:: none

..         docker volume rm quantecon
..         docker volume create quantecon

.. The first line will delete any existing volume we had with that name.

.. Usage
.. ---------------------

.. The basic command is (Linux, OS/X)

.. .. code-block:: none

..     docker run --rm -p 8888:8888 -v quantecon:/home/jovyan/.julia -v "$(pwd)":/home/jovyan/local quantecon/base

.. Or on Powershell on Windows

.. .. code-block:: none

..     docker run --rm -p 8888:8888 -v quantecon:/home/jovyan/.julia -v ${PWD}:/home/jovyan/local quantecon/base

.. * The ``rm`` instructs Docker to delete the container on exit,

.. * The ``PWD`` statement will mount the local directory (i.e., where the terminal is) to the Docker for exchanging files.

.. * The ``p`` flag is for browser access.

.. * The ``quantecon:/home/jovyan/.julia`` mount is for persisting changes we make to the Julia user depot.

.. You will see something like

.. .. figure:: /_static/figures/docker-basic-command.png
..     :width: 100%

.. In the output, you should see some text near that bottom that looks like

.. .. code-block:: none

..     127.0.0.1):8888/?token=7c8f37bf32b1d7f0b633596204ee7361c1213926a6f0a44b

.. Copy the text after ``?token=`` (e.g. ``7c8f37bf32b1d7f0b633596204ee7361c1213926a6f0a44b``).

.. In a browser, go to a URL like the following

.. .. code-block:: none

..         http://127.0.0.1:8888/lab

.. To see something like

.. .. figure:: /_static/figures/docker-jupyter-lab.png
..     :width: 100%

.. **Note**:

.. * ``Ctrl+C`` is also the keyboard shortcut you use to kill the container, so be sure to copy using the mouse.
.. * When you call this command, Docker may require you to give it permissions to access the drive and the network.  If you do not see the output within 20 or so seconds, then look for confirmation windows which may be hidden behind the terminal/etc.

.. Paste the text into ``Password or token:`` and choose ``Log in`` to get the full window

.. .. figure:: /_static/figures/docker-jlab-full.png
..     :width: 100%

.. We can see that some packages are already pre-installed for our use

.. .. figure:: /_static/figures/docker-packages-preinstalled.png
..     :width: 100%

.. Maintenance and Troubleshooting
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. A few reminders

.. * If you forget your token number, you may need to stop and restart the container.
.. * To stop the container, use ``Ctrl-C`` or type ``docker stop $(docker ps -aq)`` in a different terminal.
.. * To reset your Docker volume completely, redo the ``docker volume rm quantecon`` and ``docker volume create quantecon`` steps.
.. * To clean unnecessary Docker assets from your system, run ``docker system prune``.
.. * If you can't log in, make sure you don't have an existing jupyter notebook occupying port 8888. To check, run ``jupyter notebook list`` and (if need be) ``jupyter notebook stop 8888``. If you have difficulties, see `this git issue <https://github.com/jupyter/notebook/issues/2844>`_.
