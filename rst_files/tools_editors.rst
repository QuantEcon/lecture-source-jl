.. _tools_editors:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Julia Tools and Editors
******************************************

.. contents:: :depth: 2

Co-authored with Arnav Sood

While Jupyter notebooks are a great way to get started with the language, eventually you'll want to use more powerful Tools

We assume you've already completed the :ref:`getting started <getting_started>` lecture 

The REPL
=============

Previously, we discussed basic use of the Julia REPL ("Read-Evaluate-Print Loop")

Here, we'll consider some more advanced features

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

Hitting ``]`` brings you into package mode

* ``] add Expectations`` will add a package (here, ``Expectations.jl``) 

* Likewise, ``] rm Expectations`` will remove that package 

* ``] st`` will show you a snapshot of what you have installed 

* ``] up`` will (intelligently) upgrade versions of your packages 

* ``] precompile`` will precompile everytihng possible 

You can get a full list of package-mode commands by running 

.. code-block:: julia 

    ] ? 

Help Mode 
---------------

Hitting ``?`` will bring you into help mode 

The key use case is to find docstrings for functions and macros, e.g. 

.. code-block:: julia 

    ? print 

Note that objects must be loaded for Julia to return their documentation, e.g. 

.. code-block:: julia 

    ? @test 

will fail, but 

.. code-block:: julia 

    using Test 

.. code-block:: julia 

    ? @test 

will succeed 

Atom 
=========

As discussed `previously <getting_started>`_, eventually you'll want to use a full-fledged text editor 

The most feature-rich one for Julia development is `Atom <https://atom.io/>`_

Installation and Configuration 
---------------------------------

Instructions for basic setup and configuration can be found `here <https://github.com/econtoolkit/tutorials/blob/master/julia.md#installation-and-setup/>`_

The key package to install is called `Juno <http://junolab.org.>`_ 

Standard Layout  
------------------

If you follow the instructions, you should see something like this when you open a new file 

If you don't, simply go to the command palette and type "Julia standard layout" 

.. figure:: /_static/figures/juno-standard-layout.png
    :scale: 60%

The bottom pane is a standard REPL, which supports the different modes above 

The "workspace" pane is a snapshot of currently-defined objects. For example, if we define an object in the REPL

.. code-block:: julia 

    x = 2

Our workspace should read 

.. figure:: /_static/figures/juno-workspace-1.png
    :scale: 60%

The ``ans`` variable simply captures the result of the last computation 

The ``Documentation`` pane simply lets us query Julia documentation 

.. figure:: /_static/figures/juno-docs.png
    :scale: 60%

The ``Plots`` pane captures Julia plots output 

.. figure:: /_static/figures/juno-plots.png
    :scale: 60%

May be buggy, see for ex: `here <https://github.com/MTG/sms-tools/issues/36/>`_

Other Features 
-------------------

* `` Shift + Enter `` will evaluate a highlighted selection or line (as above)

* The run symbol in the left sidebar (or ``Ctrl+Shift+Enter``) will run the whole file 

Docker Integration 
----------------------

You can plug Juno/Atom into a Julia session running in a docker container, such as the QuantEcon base container 

For instructions on this, see the `Juno FAQ <https://docs.junolab.org/latest/man/faq.html/>_`

Package Environments 
========================

Julia's package manager lets you set up Python-style "virtualenvs," that draw from an underlying pool of assets on the machine 

* An ``environment`` is a set of packages specified by a ``Project.toml`` (and optionally, a ``Manifest.toml``) 

* A ``registry`` is a git repository corresponding to a list of (typically) registered packages, from which Julia can pull 

* A ``depot`` is a directory, like ``~/.julia``, which contains assets (compile caches, registries, package source directories, etc.) 

Essentially, an environment is a dependency tree for a project, or a "frame of mind" for Julia's package manager 

We can see the base (``v1.0``) environment as such 

.. code-block:: julia 

    ] st 

And we can switch environments as such 

.. code-block:: julia 

    using InstantiateFromURL 
    activate_github("arnavs/InstantiationTest")

.. code-block:: julia 

    ] st 

We can edit environments, as before 

.. code-block:: julia 

    ] add Distributions 

And this will propagate changes to the TOML 

.. code-block:: julia 

    Base.active_project()

Lastly, we can return to the default environment by calling ``activate`` without args 

.. code-block:: julia 

    ] activate 