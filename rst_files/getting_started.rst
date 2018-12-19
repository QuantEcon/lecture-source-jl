.. _getting_started:

.. include:: /_static/includes/lecture_howto_jl_full.raw

*************************************
Setting up Your Julia Environment
*************************************

.. contents:: :depth: 2

Overview
============

In this lecture we will cover how to get up and running with Julia

There are a few different permutations of the setup (cloud-based vs on your machine, native vs :ref:`virtual <jl_jupyterdocker>`, etc.), so some sections may not directly apply to you

We'll simultaneously introduce `Jupyter <http://jupyter.org/>`_, which is the browser-based blend of text, math, and code that these lectures are written in

A Note on Jupyter
=====================

Like Python and R, and unlike products such as Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a specific development environment

While you will eventually use other editors, there are some advantages to starting with Jupyter while learning Julia

* The ability to mix formatted text (including mathematical expressions) and code in a single document

* Nicely formatted output including tables, figures, animation, video, etc.

* Conversion tools to generate PDF slides, static HTML, etc.

* Can be used in the :ref`cloud <jl_jupyterhub>` without requiring installation

We'll discuss the workflow on these features in the :doc:`next lecture <julia_environment>`

.. _jl_jupyterlocal:

Desktop Installation of Julia and Jupyter
==============================================

If you want to install these tools locally on your machine

* Install Anaconda Python by `downloading the binary <https://www.anaconda.com/download/>`

    * Make sure you click yes to "add Anaconda to my PATH"

* Download and install Julia, from `download page <http://julialang.org/downloads/>`_ , accepting all default options

    * We do not recommend `JuliaPro <https://juliacomputing.com/products/juliapro.html>`_
      due to its limited number of available packages

.. _intro_repl:

* Open Julia, by either:

    #. Navigating to Julia through your menus or desktop icons (Windows, OSX), or

    #. Opening a terminal and typing ``julia`` (Linux)

Either way you should now be looking at something like this (modulo your operating system)

.. figure:: /_static/figures/julia_term_1.png
   :scale: 100%

This is called the JULIA *REPL* (Read-Evaluate-Print-Loop), which we discuss more :ref:`later <julia_repl>`

* In the Julia REPL, hit ``]`` to enter package mode and then enter

.. code-block:: julia
    :class: no-execute

    add IJulia

This is the ``IJulia`` kernel which links Julia to Jupyter (i.e., allows your browser to run Julia code, manage Julia packages, etc.)

* To run Jupyter, open a terminal or windows console, ``cd`` to the location you wish to write files and type

.. code-block:: none

    jupyter lab

.. _package_setup:

Package Setup
------------------

To install a curated set of packages that we use with the lectures

* Inside the Julia REPL, run

.. code-block:: julia
   :class: no-execute

   ] add InstantiateFromURL

This will install a tool written by the QE team to manage dependencies for the lectures

* Next, run

.. code-block:: julia
   :class: no-execute

   using InstantiateFromURL

This will load the functions defined in the ``InstantiateFromURL`` package

* Next, run

.. code-block:: julia
   :class: no-execute

   activate_github("QuantEcon/QuantEconLecturePackages", tag = "v0.9.5", add_default_environment = true)

This function will:

1. Download two files, ``Project.toml`` and ``Manifest.toml``, containing exact dependency information for ``v0.9.5`` of the QuantEconLecturePackages set

2. Install those packages to your machine.

3. Add them to the ``v1.0`` environment, which is what a fresh Julia instance starts from.

The last line simply means that Julia is capable of storing multiple (and even mutually inconsistent) versions of the same packages

The package manager knows which ones you mean (i.e., what to load when you type ``using ExamplePackage``) by investigating the **active environment**

An environment in Julia is simply a pair of TOML files (as above), where the ``Project.toml`` is a list of dependencies, and the ``Manifest.toml`` provides exact version information

We will cover this more in depth later in the lectures, but the upshot is that you won't need to run any further code to use these packages


Cloud-Based and Virtual Alternatives
=====================================

There are alternative workflows, such as

#. Using :ref:`Jupyter on the cloud or a department server <jl_jupyterhub>` (if it is available)
#. Installing the pre-built :ref:`docker-based Julia/Jupyter <jl_jupyterdocker>` from QuantEcon

Eventually, you will want to move from just using Jupyter to using other
:doc:`tools and editors <tools_editors>` such as `Atom/Juno <http://junolab.org/>`_, but
don't let the environment get in the way of learning the language

.. _jl_jupyterhub:

Cloud Solutions
---------------------------

If you have access to a cloud-based solution for Jupyter, then that is typically a straightforward option

* Students: ask your department if these resources are available
* Universities and workgroups: email `contact@quantecon.org <mailto:contact@quantecon.org">`_ for
  help on setting up a shared JupyterHub instance with precompiled packages ready for these lecture notes
* `JuliaBox <http://www.juliabox.com>`_  tightly controls allowed packages, and **does not** currently support the QuantEcon lectures

.. _jl_jupyterdocker:

Package Setup
^^^^^^^^^^^^^^

As :ref:`above <package_setup>`, these lectures depend on functionality (like plotting, benchmarking, and statistics) that doesn't always come with base Julia

To solve this issue, we've created two objects:

1. A curated `list of packages <https://github.com/quantecon/quanteconlecturepackages>`_ that we use in testing

2. A Julia package, ``InstantiateFromURL.jl``, for installing and loading them

Here are the steps to follow:

* Make sure that you don't have the packages we use already installed (i.e., run ``] st`` in the REPL). Many JupyterHub setups will come with a large set of pre-installed packages.

* If they aren't installed, hit ``]`` and run the following

.. code-block:: julia

   add InstantiateFromURL

* If that works, hit backspace (to get back into the main Julia REPL)

.. code-block:: julia
   :class: no-execute

   using InstantiateFromURL
   activate_github("QuantEcon/QuantEconLecturePackages", tag = "v0.9.5")

We describe what these functions do in the relevant part of the :ref:`earlier section <package_setup>`

But, at a glance, this function will download dependency information for the packages we use as a Julia "environment" (Julia can maintain multiple such "environments," which allow for mutually inconsistent package versions to be installed on the same machine)

And it will act on that dependency information; namely, making sure that the specified versions exist on the machine

* If the above two steps failed, you can try adding individual packages as needed (i.e., ``] add Expectations StatPlots NLsolve...``).

* Depending on your JupyterHub setup, this might also fail (i.e., maybe you don't have permission to install, or they lock down the packages from which you can choose). In that case, you'd need to speak to an administrator for the setup


Installing a Pre-built Jupyter Image
---------------------------------------

`Docker <https://www.docker.com/>`_ is a technology that you use to host
a "`virtual <https://en.wikipedia.org/wiki/Operating-system-level_virtualization>`_"
version of a software setup on another computer

While it is largely used for running code in the cloud, it is also convenient for using on local computers

QuantEcon has constructed a pre-built `docker image <https://hub.docker.com/u/quantecon/>`_

For instructions on how to use it, see the :doc:`tools and editors <tools_editors>` lecture

**Note:** The Docker installation is easy and complete, but it has limitations
on operating systems (in particular, Windows 10 is only supported for the Professional
and Education editions, and not the Home edition)

Package Setup
^^^^^^^^^^^^^^^^^

One of the key advantages of the Docker is that all the setup steps are baked in

As a result, these images come with the full set of packages out-of-the-box
