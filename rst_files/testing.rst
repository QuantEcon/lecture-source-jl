.. _testing:

.. include:: /_static/includes/lecture_howto_jl.raw

***************************************************
Packages, Testing, and Continuous Integration 
***************************************************

Co-authored with Arnav Sood

.. contents:: :depth: 2

This lecture is about structuring your project as a Julia module, and testing it with tools from GitHub 

Benefits include 

* Specifying dependencies (and their versions), so that your project works across Julia setups and over time

* Being able to load your project's functions from outside without copy/pasting 

* Writing tests that run locally, *and automatically on the GitHub server* 

* Having GitHub test your project across operating systems, Julia versions, etc.

Project Setup 
=======================

Online Setup 
--------------------

Julia Setup 
--------------------

.. literalinclude:: /_static/includes/deps.jl

We also want to add the `PkgTemplates <https://github.com/invenia/PkgTemplates.jl/>`_ package 

.. code-block:: julia 

    using Pkg 
    pkg"add PkgTemplates"
    pkg"precompile"

Next, let's create a *template* for our project 

This specifies metadata like the license we'll be using (MIT by default), the location (``~/.julia/dev`` by default), etc.

.. code-block:: julia 

    using PkgTemplates 
    ourTemplate = Template(;user="quanteconuser", plugins = [TravisCI(), CodeCov()])

Let's create a specific project based off this template

.. code-block:: julia 

    generate("ExamplePackage.jl", ourTemplate)

If we navigate to the package directory (shown in the output), we should see something like 

.. figure:: /_static/figures/testing-dir.png
    :scale: 60%

Adding Project to Git 
------------------------

The next step is to add this project to Git version control 

First, open the repository screen in your account as discussed previously. We'll want the following settings 

.. figure:: /_static/figures/testing-git1.png
    :scale: 60%

In particular 

* The repo you create should have the same name as the project we added 

* We should leave the boxes unchecked for the ``README.md``, ``LICENSE``, and ``.gitignore``, since these are handled by ``PkgTemplates``

Then, drag and drop your folder from your ``~/.julia/dev`` directory to GitHub Desktop 

Click the "publish branch" button to upload your files to GitHub 

If you navigate to your git repo (ours is `here <https:https://github.com/quanteconuser/ExamplePackage.jl/>`_), you should see something like 


.. figure:: /_static/figures/testing-git2.png
    :scale: 60%

Adding Project to Julia Package Manager 
-------------------------------------------

We also want Julia's package manager to be aware of the project

In the ``ExamplePackage.jl`` directory, open a new terminal and run 

.. code-block:: julia 

    using Pkg 
    pkg"dev ." 

Now, from any Julia terminal on our machine, we can run 

.. code-block:: julia 

    using Pkg 
    pkg"activate ExamplePackage" 

To work with our project, and 

.. code-block:: julia 

    using ExamplePackage

To use it 

Project Structure 
==========================

Let's unpack the structure of the generated project 

* The first directory, ``.git``, holds the version control information 

* The ``src`` directory contains the project's source code. Currently, it should contain only one file (``ExamplePackage.jl``), which reads 

.. code-block:: none 

    module ExamplePackage

    greet() = print("Hello World!")

    end # module

* Likewise, the ``test`` directory should have only one file (``runtests.jl``), which reads:

.. code-block:: none 

    using ExamplePackage
    using Test

    @testset "ExamplePackage.jl" begin
        # Write your own tests here.
    end

In particular, the workflow is to export objects we want to test (``using ExamplePackage``), and test them using Julia's ``Test`` module 

The other important text files for now are 

* ``Project.toml`` and ``Manifest.toml``, which contain dependency information 

* The ``.gitignore`` file (which may display as an untitled file), which contains files and paths for ``git`` to ignore 

Project Workflow
=========================

Dependency Management 
----------------------------

If you run 

.. code-block:: julia 

    pkg"st"

You'll notice that the our project is now the "active environment" 

This means that any dependencies we add, or package operations we execute, will be reflected in our ``ExamplePackage.jl`` directory's TOML 

This allows us to share the project with others, who can exactly reproduce the state used to build and test it 

See the `Pkg3 docs <https://docs.julialang.org/en/v1/stdlib/Pkg/>`_ for more information 

For now, let's try adding a dependency 

.. code-block:: julia 

    pkg"add Expectations"

Our ``Project.toml`` should now read something like::

    name = "ExamplePackage"
    uuid = "f85830d0-e1f0-11e8-2fad-8762162ab251"
    authors = ["QuantEcon User <quanteconuser@gmail.com>"]
    version = "0.1.0"

    [deps]
    Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"

    [extras]
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

    [targets]
    test = ["Test"]

The ``Manifest.toml`` (which tracks exact versions) has changed as well, to include a list of sub-dependencies and versions 

.. figure:: /_static/figures/testing-atom-manifest.png
    :scale: 60%

Writing Code
-----------------

The basic idea is to work in ``tests/runtests.jl``, while reproducible functions should go in the ``src/ExamplePackage.jl``

For example, let's say we add ``Distributions.jl`` and edit the source to read as follows::

    module ExamplePackage

    greet() = print("Hello World!")

    using Expectations, Distributions

    function foo(μ = 1., σ = 2.)
        d = Normal(μ, σ)
        E = expectation(d)
        return E(x -> sin(x))
    end

    export foo 

    end # module

Let's try calling this from a fresh Julia REPL::

    julia> using ExamplePackage
    [ Info: Recompiling stale cache file C:\Users\Arnav Sood\.julia\compiled\v1.0\ExamplePackage\hpt8s.ji for ExamplePackage [f85830d0-e1f0-11e8-2fad-8762162ab251]

    julia> foo()
    0.11388071406436832

Jupyter Workflow 
------------------------

We can also call this function from a Jupyter notebook 

Let's create a new output directory in our project, and run ``jupyter lab`` from it. Call a new notebook ``output.ipynb``

.. figure:: /_static/figures/testing-output.png
    :scale: 60%

From here, we can use our package's functions in the usual way. This lets us produce neat output examples, without re-defining everything 

We can also edit it interactively inside the notebook 

.. figure:: /_static/figures/testing-notebook.png
    :scale: 60%

The change will be reflected in the ``Project.toml`` file::

    name = "ExamplePackage"
    uuid = "f85830d0-e1f0-11e8-2fad-8762162ab251"
    authors = ["QuantEcon User <quanteconuser@gmail.com>"]
    version = "0.1.0"

    [deps]
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Expectations = "2fe49d83-0758-5602-8f54-1f90ad0d522b"
    Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"

    [extras]
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

    [targets]
    test = ["Test"]

And the Manifest as well 

Be sure to add ``output/.ipynb_checkpoints`` to your ``.gitignore`` file, so that's not checked in 

Collaborative Work 
--------------------------

For someone else to get the package, they simply need to run 

.. code-block:: julia 

    using Pkg 
    pkg"dev https://github.com/quanteconuser/ExamplePackage.jl.git"

This will place the repository inside their ``~/.julia/dev`` folder, and they can drag-and-drop it to GitHub desktop in the usual way 

They can then collaborate as they would on other git repositories 

Unit Testing
====================================

It's important to make sure that your code is well-tested

There are a few different kinds of test, each with different purposes

#. *Unit testing* makes sure that individual pieces of a project function as expected

#. *Integration testing* makes sure that they work together as expected 

#. *Regression testing* makes sure that behavior is unchanged over time 

In this lecture, we'll focus on unit testing 

The ``Test`` Module
-------------------------



Continuous Integration with Travis
==========================================

TODO 

CodeCoverage 
===================

TODO 

Benchmarking 
==================

TODO 