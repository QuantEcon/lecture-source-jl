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

That said, you **should still read through them**, as there is useful information throughout

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

    add IJulia InstantiateFromURL

* Then, hit backspace to come into the main REPL mode and enter the following

    using InstantiateFromURL
    activate_github("QuantEcon/QuantEconLecturePackages", tag = "v0.9.5", add_default_environment = true)

This installs enough Julia packages to begin using the lecture notes

Depending on your computer, this may take **10-15 minutes** to run the **first-time**, but be virtually instantaneous thereafter

More precisely:

1. The QuantEcon lectures depend on outside pieces of code called packages, which are standalone code bundles that provide specific functionality (we'll discuss some :doc:`useful ones <general_packages>` later on)

2. The first line installs the ``IJulia`` package (which connects Jupyter to Julia), and the ``InstantiateFromURL`` package, which is a tool the QE team produced to manage dependencies for the lectures

3. The second line says that we want to use functionality from that second tool

4. The third line will grab a curated set of packages from the ``QuantEcon/QuantEconLecturePackages`` GitHub repository (we talk about GitHub in :doc:`version control <version_control>`), install them on the machine, and also add them to the default ``v1.0`` environment so we can use them right away (for more on Julia environments, see :doc:`tools and editors <tools_editors>`)

* To run Jupyter, open a terminal or windows console, ``cd`` to the location you wish to write files and type

.. code-block:: none

    jupyter lab

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

If you're using a cloud setup, you may not need to do anything to begin using these notebooks

Otherwise, if you are missing packages, you will need to go open a Jupyter notebook and type

.. code-block:: julia
    :class: no-execute

    ] add InstantiateFromURL

If you see an error, then your online JupyterHub may not support adding new packages programmatically

In this case, you can try adding new packages manually as-needed (i.e., by typing ``] add Package1 Package2 Package3...`` into its own cell)

If that doesn't work, though, you may need to contact the administrator of your cloud setup

Assuming that the above command was successful, run

.. code-block:: julia

    using InstantiateFromURL
    activate_github("QuantEcon/QuantEconLecturePackages", tag = "v0.9.5")

This is similar to the above, except it doesn't try to propagate changes to the default Julia environment (which may be impossible due to permissions, etc.)

This means that you will have to run these lines (i.e., "activate" the QuantEcon environment) whenever you want to use the packages in the curated set

.. _jl_jupyterdocker:

Installing a Pre-built Jupyter Image
---------------------------------------

`Docker <https://www.docker.com/>`_ is a technology that you use to host
a "`virtual <https://en.wikipedia.org/wiki/Operating-system-level_virtualization>`_"
version of a software setup on another computer

While it is largely used for running code in the cloud, it is also convenient for using on local computers

QuantEcon has constructed a pre-built `docker image <https://hub.docker.com/u/quantecon/>`_

For instructions on how to set this up, see the :doc:`tools and editors <tools_editors>` lecture

**Note:** The Docker installation is easy and complete, but it has limitations
on operating systems (in particular, Windows 10 is only supported for the Professional
and Education editions, and not the Home edition)
