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
==================================

Git Setup 
----------------

First, you need to configure your global GitHub information 

You can do this by running the following in the terminal 

.. code-block:: none 

    git config --global user.name "QuantEcon User" # your real name goes here 
    git config --global user.email "quanteconuser@gmail.com" # your github email goes here 
    git config --global github.user "quanteconuser" # your github username goes here 

Next, we want to make a new project from the Julia console 

Let's load our Julia environment 

Julia Setup 
--------------------

.. literalinclude:: /_static/includes/deps.jl

We also want to add the `PkgTemplates <https://github.com/invenia/PkgTemplates.jl/>`_ package 

.. code-block:: julia 

    using Pkg 
    pkg"add PkgTemplates"
    pkg"precompile"

Template Creation 
-----------------------

Next, let's create a *template* for our project 

This specifies metadata like the license we'll be using (MIT by default), the location (``~/.julia/dev`` by default), etc.

.. code-block:: julia 

    using PkgTemplates 
    ourTemplate = Template(;plugins = [TravisCI(), CodeCov()])

You should see the following output

.. code-block:: none 

    julia> ourTemplate = Template(;plugins = [TravisCI(), CodeCov()])
Template:
  → User: quanteconuser
  → Host: github.com
  → License: MIT (QuantEcon User 2018)
  → Package directory: ~\.julia\dev
  → Minimum Julia version: v1.0
  → SSH remote: No
  → Plugins:
    • CodeCov:
      → Config file: None
      → 3 gitignore entries: "*.jl.cov", "*.jl.*.cov", "*.jl.mem"
    • TravisCI:
      → Config file: Default
      → 0 gitignore entries

Next, let's create a specific project based off this template

.. code-block:: julia 

    generate("ExamplePackage.jl", ourTemplate)

If we navigate to ``~/.julia/dev`` (you can find the location of ``.julia`` by running ``Sys.BINDIR``), you should a directory like

.. figure:: /_static/figures/testing-dir.png
    :scale: 60%

Adding Project to Git 
---------------------------------

The next step is to add this project to Git version control 

First, open the repository screen in your account as discussed previously. We'll want the following settings 

.. figure:: /_static/figures/testing-gitrepo.png
    :scale: 60%

In particular 

* The repo you create should have the same name as the project we added 

* We should leave the boxes unchecked for the ``README.md``, ``LICENSE``, and ``.gitignore``, since these are handled by ``PkgTemplates``

Then, drag and drop your folder from your ``~/.julia/dev`` directory to GitHub Desktop 

Click the "publish branch" button to upload your files to GitHub 

If you navigate to your git repo (ours is `here <https:https://github.com/quanteconuser/ExamplePackage.jl/>`_), you should see something like 

.. figure:: /_static/figures/testing-gitrepo2.png
    :scale: 60%

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

Unit Testing Frameworks
====================================

TODO

Continuous Integration with Travis
==========================================

It's now time to enable continuous integration, so that GitHub runs these tests for us 

The tool we'll use for this is called `Travis CI <https://travis-ci.org/>`_

Travis Setup 
-----------------------

If you log on to the Travis site and are logged into GitHub in the same browser session, you should see a pane like 

.. figure:: /_static/figures/testing-travis-setup.png
    :scale: 60%

Click authorize 

You should see a list of all your repositories, as follows 

.. figure:: /_static/figures/testing-travis-repo-list.png
    :scale: 60%

To enable continuous integration on a repo, simply click the grey slider to move it to the right 

You can then click the repo name to the left of the slider to get to the Travis page for the repo. Ours should look something like 

.. figure:: /_static/figures/testing-travis-repo-page.png
    :scale: 60%

Code Coverage 
==========================

It's also important to be aware of *how much* of your code is covered by unit tests 

To do this, we can enable the code coverage repo

CodeCov Setup 
------------------------------

First, navigate to the `CodeCov website <https://codecov.io/>` and hit "sign up with GitHub":

.. figure:: /_static/figures/testing-codecov-signup.png
    :scale: 60%

Clicking "authorize" on the resulting window should bring you to a screen like 

.. figure:: /_static/figures/testing-codecov-signup-2.png
    :scale: 60%

Click "add repository," and then hit "add private scope" on the next page to allow CodeCov to plug in to your private projects

Now, we can click through "add repository" again, and hit our repository name, to give us a screen like 

.. figure:: /_static/figures/testing-codecov-token.png
    :scale: 60%

Next, we'll need to go to our ``.travis.yml`` file. You'll notice at the bottom it has something like

.. code-block:: none 

    after_success:
        - julia -e 'using Pkg; Pkg.add("Coverage"); using Coverage; Codecov.submit(process_folder())'

If the repo is public, this is all we need --- no changes are necessary 

If the repo is private, we will need to set the token as a Travis environment variable 

In a new tab, go to our repo's travis page and hit "settings" under "more options"

.. figure:: /_static/figures/testing-travis-env.png
    :scale: 60%

Add our token as below 

.. figure:: /_static/figures/testing-travis-token.png
    :scale: 60%

Our repo is now configured to have Travis call to CodeCov after its tests are complete

Benchmarking
=======================

TODO [see `need for speed`, basically]