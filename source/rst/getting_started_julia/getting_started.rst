.. _getting_started:

.. include:: /_static/includes/header.raw

*************************************
Setting up Your Julia Environment
*************************************

.. contents:: :depth: 2

Overview
============

In this lecture we will cover how to get up and running with Julia

There are a few different options for using Julia, including a :ref:`local desktop installation <jl_jupyterlocal>` and :ref:`Jupyter hosted on the web<jl_jupyterhub>`

If you have access to a web-based Jupyter and Julia setup, it is typically the most straightforward way to get started

A Note on Jupyter
=====================

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment

While you will eventually use other editors, there are some advantages to starting with the `Jupyter <http://jupyter.org/>`_ environment while learning Julia

* The ability to mix formatted text (including mathematical expressions) and code in a single document

* Nicely formatted output including tables, figures, animation, video, etc.

* Conversion tools to generate PDF slides, static HTML, etc.

* :ref:`Online Jupyter <jl_jupyterhub>` may be available, and requires no installation

We'll discuss the workflow on these features in the :doc:`next lecture <../getting_started_julia/julia_environment>`

.. _jl_jupyterlocal:

Desktop Installation of Julia and Jupyter
==============================================

If you want to install these tools locally on your machine

* Download and install Julia, from `download page <http://julialang.org/downloads/>`_ , accepting all default options

    * We do **not** recommend `JuliaPro <https://juliacomputing.com/products/juliapro.html>`_

.. _intro_repl:

* Open Julia, by either

    #. Navigating to Julia through your menus or desktop icons (Windows, Mac), or

    #. Opening a terminal and typing ``julia`` (Linux)

You should now be looking at something like this

.. figure:: /_static/figures/julia_term_1.png
   :scale: 100%

This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more :ref:`later <repl_main>`

* In the Julia REPL, hit ``]`` to enter package mode and then enter

.. code-block:: julia
    :class: no-execute

    add IJulia InstantiateFromURL

This adds packages for

* The  ``IJulia`` kernel which links Julia to Jupyter (i.e., allows your browser to run Julia code, manage Julia packages, etc.)

* The ``InstantiateFromURL`` which is a tool written by the QE team to manage package dependencies for the lectures

.. _jupyter_installation:

Installing Jupyter
-------------------

If you have previously installed Jupyter (e.g., installing Anaconda Python by `downloading the binary <https://www.anaconda.com/download/>`)
then the ``add IJulia`` installs everything you need into your existing environment

Otherwise - or in addition - you can install it directly from the Julia REPL

.. code-block:: julia
    :class: no-execute

    using IJulia; jupyterlab(detached = true)

Choose the default, ``y`` when asked to install Jupyter and then JupyterLab via Conda

After the installation, a JupyterLab tab should open in your browser

(Optional) To enable launching JupyterLab from a terminal, use :ref:`add Julia's Jupyter to your path <add_jupyter_to_path>`

.. _clone_lectures:

Starting Jupyter
------------------

Next, let's install the QuantEcon lecture notes to our machine and run them (for more details on the tools we'll use, see our lecture on :doc:`version control <../more_julia/version_control>`)

1. Install `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/>`_

2. (**Optional, but strongly recommended**) Install the `GitHub Desktop <https://desktop.github.com/>`_

GitHub Desktop Approach
^^^^^^^^^^^^^^^^^^^^^^^^^

After installing the Git Desktop application, click `this link <x-github-client://openRepo/https://github.com/QuantEcon/quantecon-notebooks-julia>`_ on your desktop computer to automatically install the notebooks

It should open a window in the GitHub desktop app like this

.. figure:: /_static/figures/git-desktop-intro.png
   :scale: 100%

Choose a path you like and clone the repo

**Note:** the workflow will be easiest if you clone the repo to the default location relative to the home folder for your user

From a Julia REPL, start JupyterLab by executing

.. code-block:: julia
    :class: no-execute

    using IJulia; jupyterlab(detached = true)


Alternatively, if you installed Jupyter separately in :ref:` Jupyter Installation <jupyter_installation>` or :ref:`added Jupyter to your path <add_jupyter_to_path>` then run ``jupyter lab`` in your terminal

.. At the top, under the "Repository" dropdown, click "Open in Terminal" (Mac, Linux) or "Open in Command Prompt" (Windows)

.. **Note**: On Windows, you may need to click the "open without git" button that comes up
..
.. In the resulting terminal session, run
..
.. .. code-block:: none
..
..     jupyter lab
..

Navigate to the location you stored the lecture notes, and open the :doc:`Interacting with Julia <../getting_started_julia/julia_environment>` notebook to explore this interface and start writing code

Git Command Line Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not wish to install the GitHub Desktop, you can get the notebooks using the Git command-line tool

Open a new terminal session and run

.. code-block:: none

    git clone https://github.com/quantecon/quantecon-notebooks-julia

This will download the repository with the notebooks in the working directory

Then, ``cd`` to that location in your Mac, Linux, or Windows PowerShell terminal

.. code-block:: none

    cd quantecon-notebooks-julia

Then, either using the ``using IJulia; jupyterlab(detached = true)`` or execute ``jupyter lab`` within your shell

And open the :doc:`Interacting With Julia <../getting_started_julia/julia_environment>` lecture (the file ``julia_environment.ipynb`` in the list of notebooks in JupyterLab) to continue

Using Julia on the Web
=====================================

If you have access to an online Julia installation, it is the easiest way to get started

Eventually, you will want to do a :ref:`local installation <jl_jupyterlocal>` in order to use other
:doc:`tools and editors <../more_julia/tools_editors>` such as `Atom/Juno <http://junolab.org/>`_, but
don't let the environment get in the way of learning the language

.. _jl_jupyterhub:

Using Julia with JupyterHub
----------------------------

If you have access to a web-based solution for Jupyter, then that is typically a straightforward option

* Students: ask your department if these resources are available
* Universities and workgroups: email `contact@quantecon.org <mailto:contact@quantecon.org">`_ for
  help on setting up a shared JupyterHub instance with precompiled packages ready for these lecture notes
* `JuliaBox <http://www.juliabox.com>`_  tightly controls allowed packages, and **does not** currently support the QuantEcon lectures

Obtaining Notebooks
^^^^^^^^^^^^^^^^^^^^^

Your first step is to get a copy of the notebooks in your JupyterHub environment

While you can individually download the notebooks from the website, the easiest way to access the notebooks is usually to clone the repository with Git into your JupyterHub environment

JupyterHub installations have different methods for cloning repositories, with which you can use the url for the notebooks repository: ``https://github.com/QuantEcon/quantecon-notebooks-julia``

.. The left side of JupyterHub's interface has a ``files`` pane which you can use to navigate to and open the lectures (more on this in the next lecture)

.. _package_setup: 

Installing Packages
=====================

After you have some of the notebooks available, as in :ref:`above <clone_lectures>`, these lectures depend on functionality (like packages for plotting, benchmarking, and statistics) that are not installed with every Jupyter installation on the web

If your online Jupyter does not come with QuantEcon packages pre-installed, you can install the ``InstantiateFromURL`` package, which is a tool written by the QE team to manage package dependencies for the lectures

.. Not sure this is needed in the workflow, and this does not give enough perspective to explain how to use it.
.. * Make sure that you don't have the packages we use already installed (i.e., run ``] st`` in the REPL). Many JupyterHub setups will come with a large set of pre-installed packages.

To add this package, in an online Jupyter notebook run (typically with ``<Shift-Enter>``)

.. code-block:: julia
    :class: hide-output

    ] add InstantiateFromURL

If your online Jupyter environment does not have the packages pre-installed, it may take 15-20 minutes for your first QuantEcon notebook to run

After this step, open the downloaded :doc:`Interacting with Julia <../getting_started_julia/julia_environment>` notebook to begin writing code

If the QuantEcon notebooks do not work after this installation step, you may need to speak to the JupyterHub administrator

.. JuliaBox
.. --------------------------
.. TODO WHEN JuliaBox works, add this in
.. JuliaBox is a particular JupyterHub installation
.. How to install the packages on julia
.. How to do a git pull
.. Point out that you do not do the `activate_github` step above....


.. NOT NECESARY FOR NOW.  The lack of support for basic Windows means this could confuse some users
.. Installing a Pre-built Jupyter Image
.. ---------------------------------------
..
.. `Docker <https://www.docker.com/>`_ is a technology that you use to host
.. a "`virtual <https://en.wikipedia.org/wiki/Operating-system-level_virtualization>`_"
.. version of a software setup on another computer
..
.. While it is largely used for running code in the cloud, it is also convenient for using on local computers
..
.. QuantEcon has constructed a pre-built `docker image <https://hub.docker.com/u/quantecon/>`_
..
.. For instructions on how to use it, see the :doc:`tools and editors <tools_editors>` lecture

.. **Note:** The Docker installation is easy and complete, but it has limitations
.. on operating systems (in particular, Windows 10 is only supported for the Professional
.. and Education editions, and not the Home edition)

.. Package Setup
.. ^^^^^^^^^^^^^^^^^
..
.. One of the key advantages of the Docker is that all the setup steps are baked in
..
.. As a result, these images come with the full set of packages out-of-the-box
