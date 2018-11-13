.. _testing:

.. include:: /_static/includes/lecture_howto_jl_full.raw

***************************************************
Packages, Testing, and Continuous Integration 
***************************************************

.. contents:: :depth: 2

Co-authored with Arnav Sood

This lecture is about structuring your project as a Julia module, and testing it with tools from GitHub 

Benefits include 

* Specifying dependencies (and their versions), so that your project works across Julia setups and over time

* Being able to load your project's functions from outside without copy/pasting 

* Writing tests that run locally, *and automatically on the GitHub server* 

* Having GitHub test your project across operating systems, Julia versions, etc.

.. _project_setup:

Project Setup 
=======================

Account Setup 
--------------------

Travis CI 
^^^^^^^^^^^^^

As we'll see later, Travis is a service that automatically tests your project on the GitHub server 

First, we need to make sure that your GitHub account is set up with Travis CI and CodeCov 

As a reminder, make sure you signed up for the GitHub `Student Developer Pack <https://education.github.com/pack/>`_ or `Academic Plan <https://help.github.com/articles/applying-for-an-academic-research-discount/>_` if eligible 

Navigate to the `Travis website <https://travis-ci.com/>`_ and click "sign up with GitHub." Supply your credentials 

If you get stuck, see the `Travis tutorial <https://docs.travis-ci.com/user/tutorial/>`_

**NOTE** As of May 2018, Travis is deprecating the ``travis-ci.org`` website. All users should use ``travis-ci.com``

CodeCov 
^^^^^^^^^

CodeCov is a service that tells you how expansive your tests are (i.e., how much of your code is untested)

To sign up, visit the `CodeCov website <http://codecov.io/>`_, and click "sign up." You should see something like this 

.. figure:: /_static/figures/codecov-1.png
    :scale: 45%

Next, click "add a repository" and *enable private scope* (this allows CodeCov to service your private projects)

The result should be 

.. figure:: /_static/figures/codecov-2.png
    :scale: 45%

This is all we need for now 

Julia Setup 
--------------------

.. literalinclude:: /_static/includes/alldeps.jl

We also want to add the `PkgTemplates <https://github.com/invenia/PkgTemplates.jl/>`_ package 

.. code-block:: julia 

    ] add PkgTemplates
    
.. code-block:: julia 
    
    ] precompile 

To recall, you can get into the ``pkg>`` mode by hitting ``]`` in the REPL 

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
    :scale: 45%

Adding Project to Git 
------------------------

The next step is to add this project to Git version control 

First, open the repository screen in your account as discussed previously. We'll want the following settings 

.. figure:: /_static/figures/testing-git1.png
    :scale: 45%

In particular 

* The repo you create should have the same name as the project we added 

* We should leave the boxes unchecked for the ``README.md``, ``LICENSE``, and ``.gitignore``, since these are handled by ``PkgTemplates``

Then, drag and drop your folder from your ``~/.julia/dev`` directory to GitHub Desktop 

Click the "publish branch" button to upload your files to GitHub 

If you navigate to your git repo (ours is `here <https:https://github.com/quanteconuser/ExamplePackage.jl/>`_), you should see something like 

.. figure:: /_static/figures/testing-git2.png
    :scale: 45%

Adding Project to Julia Package Manager 
-------------------------------------------

We also want Julia's package manager to be aware of the project

First, open a REPL in the newly created project directory, either by noting the path printed above, or by running

.. code-block:: julia 

    DEPOT_PATH  

And navigating to the first element, then the subdirectory ``/dev/ExamplePackage``

Note the lack of ``.jl``!

You can change the path of a Julia REPL by running 

.. code-block:: julia 

    cd(joinpath(DEPOT_PATH[1], "dev", "ExamplePackage"))

Then, run 

.. code-block:: julia 

    ] activate 

To get into the main Julia environment, and 

.. code-block:: julia 

    ] dev . 

We see the change reflected in our default package list 

.. code-block:: julia 

    ] st 

Using the Package Manager 
--------------------------------

Now, from any Julia terminal in the future, we can run 

.. code-block:: none 

    using ExamplePackage

To use its exported functions

We can also get the path to this by running 

.. code-block:: none 

    using ExamplePackage
    pathof(ExamplePackage) # returns path to src/ExamplePackage.jl 

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

Environments 
^^^^^^^^^^^^^^^^

As before, the TOML files define an *environment* for our project 

Any package operations we execute will be reflected in our ``ExamplePackage.jl`` directory's TOML 

Likewise, the only packages Julia knows about are those in the ``ExamplePackage.jl`` TOML. For example

.. code-block:: julia 

    ] activate ExamplePackage

.. code-block:: julia 

    using QuantEcon # fails, even though QuantEcon is on the machine 

This allows us to share the project with others, who can exactly reproduce the state used to build and test it 

See the `Pkg3 docs <https://docs.julialang.org/en/v1/stdlib/Pkg/>`_ for more information 

Pkg Operations
^^^^^^^^^^^^^^^^^^^

For now, let's just try adding a dependency 

.. code-block:: julia 

    ] add Expectations

We can track changes in the TOML, as before

Here's the Manifest 

.. figure:: /_static/figures/testing-atom-manifest.png
    :scale: 45%

We can also run other operations, like ``] up``, ``] precompile``, etc.

Recall that, to quit the active environment and return to the base ``(v1.0)``, simply run 

.. code-block:: julia 

    ] activate 

Without any arguments

Writing Code
-----------------

The basic idea is to work in ``tests/runtests.jl``, while reproducible functions should go in the ``src/ExamplePackage.jl``

For example, let's say we add ``Distributions.jl`` 

.. code-block:: julia 

    ] activate ExamplePackage 

.. code-block:: julia 

    ] add Distributions 

and edit the source to read as follows::

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

Let's try calling this from the top environment (to stand in for a fresh REPL)

.. code-block:: julia 

    ] activate 

.. code-block:: julia 

    using ExamplePackage 
    ExamplePackage.greet()

.. code-block:: julia 

    foo() # exported, so don't need to qualify the namespace

Jupyter Workflow 
------------------------

We can also call this function from a Jupyter notebook 

Let's create a new output directory in our project, and run ``jupyter lab`` from it. Call a new notebook ``output.ipynb``

.. figure:: /_static/figures/testing-output.png
    :scale: 45%

From here, we can use our package's functions in the usual way. This lets us produce neat output examples, without re-defining everything 

We can also edit it interactively inside the notebook 

.. figure:: /_static/figures/testing-notebook.png
    :scale: 45%

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

Make sure you've activated the project environment (``] activate ExamplePackage``) before you try to propagate changes

Collaborative Work 
--------------------------

For someone else to get the package, they simply need to run 

.. code-block:: julia 

    ] dev https://github.com/quanteconuser/ExamplePackage.jl.git

This will place the repository inside their ``~/.julia/dev`` folder, and they can drag-and-drop it to GitHub desktop in the usual way 

They can then collaborate as they would on other git repositories 

In particular, they can run 

.. code-block:: julia 

    ] activate ExamplePackage 

.. code-block:: julia 

    ] instantiate 

To make sure the right dependencies are installed on their machine 

Unit Testing
====================================

It's important to make sure that your code is well-tested

There are a few different kinds of test, each with different purposes

* *Unit testing* makes sure that individual pieces of a project function as expected

* *Integration testing* makes sure that they work together as expected 

* *Regression testing* makes sure that behavior is unchanged over time 

In this lecture, we'll focus on unit testing 


The ``Test`` Module
-------------------------

Julia provides testing features through a built-in package called ``Test``, which we get by ``using Test`` 

The basic object is the macro ``@test`` 

.. code-block:: julia 

    using Test 
    @test 1 == 1 
    @test 1 ≈ 1 

Tests will pass if the condition is ``true``, or fail otherwise 

If a test is failing, we should *flag it and move on* 

.. code-block:: julia 

    @test_broken 1 == 2 

This way, we still have access to information about the test, instead of just deleting it or commenting it out 

There are other test macros, that check for things like error handling and type-stability. Advanced users can check the `Julia docs <https://docs.julialang.org/en/v1/stdlib/Test/>`_

Example 
-----------

Let's add some unit tests for the ``foo()`` function we defined earlier. Our ``tests/runtests.jl`` file should look like this 

.. code-block:: julia 

    using ExamplePackage
    using Test

    @test foo() == 0.11388071406436832
    @test foo(1, 1.5) == 0.2731856314283442
    @test_broken foo(1, 0) # tells us this is broken

And run it by running 

.. code-block:: julia 

    ] test 

from an activated REPL 

Test Sets 
-------------

By default, the ``runtests.jl`` folder starts off with a ``@testset``

This is useful for organizing different batches of tests, but for now we can simply ignore it 

To learn more about test sets, see `the docs <https://docs.julialang.org/en/v1/stdlib/Test/index.html#Working-with-Test-Sets-1/>`_

Running Tests Locally 
-----------------------

There are a few different ways to run the tests for your package 

* From a fresh REPL, run ``pkg> test ExamplePackage``

* From an activated REPL, simply run ``pkg> test`` (recall that you can activate with ``pkg> activate ExamplePackage``)

* Hit shift-enter in Atom on the actual ``runtests.jl`` file (as below). Recall that we can get the path of the package by running ``using ExamplePackage; pathof(ExamplePackage)`` from an unactivated REPL


Continuous Integration with Travis
==========================================

Setup 
-------

By default, Travis should have access to all your repositories and deploy automatically 

This includes private repos if you're on a student developer pack or an academic plan (Travis detects this automatically)

To change this, go to "settings" under your GitHub profile 

.. figure:: /_static/figures/git-settings.png
    :scale: 45%

Click "Applications," then "Travis CI," then "Configure," and choose the repos you want to be tracked 

Build Options 
----------------

By default, Travis will compile and test your project (i.e., "build" it) for new commits and PRs for every tracked repo with a ``.travis.yml`` file 

We can see ours by opening it in Atom 

.. code-block:: julia 

    # Documentation: http://docs.travis-ci.com/user/languages/julia/
    language: julia
    os:
    - linux
    - osx
    julia:
    - 1.0
    - nightly
    matrix:
    allow_failures:
        - julia: nightly
    fast_finish: true
    notifications:
    email: false
    after_success:
    - julia -e 'using Pkg; Pkg.add("Coverage"); using Coverage; Codecov.submit(process_folder())'

This is telling Travis to build the project in Julia, on OSX and Linux, using Julia v1.0 and the latest ("nightly")

It also says that if the nightly version doesn't work, that shouldn't register as a failure 

**Note** You won't need OSX unless you're building something Mac-specific, like iOS or Swift. You can delete those lines to speed up the build. Likewise for nightly Julia 

Working with Builds 
--------------------

As above, builds are triggered whenever we push changes or open a pull request 

For example, if we push our changes to the server and then click the Travis badge on the README, we should see something like 

.. figure:: /_static/figures/travis-progress.png
    :scale: 45%

This gives us an overview of all the builds running for that commit 

To inspect a build more closely (say, if it fails), we can click on it and expand the log options 

.. figure:: /_static/figures/travis-log.png
    :scale: 45%

Note that the build times here aren't informative, because we can't generally control the hardware to which our job is allocated 

We can also cancel specific jobs, either from their specific pages or by clicking the grey "x" button on the dashboard 

Lastly, we can trigger builds manually (without a new commit or PR) from the Travis overview 

.. figure:: /_static/figures/travis-trigger.png
    :scale: 45%

To commit *without* triggering a build, simply add [ci skip] somewhere inside the commit message 

Travis and Pull Requests 
----------------------------

One key feature of Travis is the ability to see at-a-glance whether PRs pass tests before merging them 

This happens automatically when Travis is enabled on a repository 

For an example of this feature, see `this PR <https://github.com/QuantEcon/Games.jl/pull/65/>`_ in the Games.jl repository 

CodeCoverage 
===================

Beyond the success or failure of our test suite, we also want to know how much of our code the tests cover 

The tool we use to do this is called `CodeCov <http://codecov.io>`_

Setup 
---------

You'll find that codecov is automatically enabled for public repos with Travis 

For private ones, you'll need to first get an access token 

Add private scope in the CodeCov website, just like we did for Travis

Navigate to the repo settings page (i.e., ``https://codecov.io/gh/quanteconuser/ExamplePackage.jl/settings`` for our repo) and copy the token 

Next, go to your travis settings and add an environment variable as below 

.. figure:: /_static/figures/travis-settings.png
    :scale: 45%

Interpreting Results 
------------------------

Click the CodeCov badge to see the build page for your project 

This shows us that our tests cover 50 \% of our functions in ``src//``

To get a more granular view, we can click the ``src//`` and the resultant filename

.. figure:: /_static/figures/codecov.png
    :scale: 45%

This shows us precisely which methods (and parts of methods) are untested 

Pull Requests to Julia Packages 
=====================================

As mentioned in ``version control``, sometimes we'll want to work on external repos that are also Julia projects 

The first thing is to ``] dev`` the git URL, which will both clone the git repo and sync it with the Julia package manager

Benchmarking 
==================

Another goal of testing is to make sure that code doesn't slow down significantly from one version to the next 

We can do this using tools provided by the ``BenchmarkTools.jl`` package 

See the ``need for speed`` lecture for more details 

Additional Notes 
=======================

* The `JuliaCI <https://github.com/JuliaCI/>`_ organization provides more Julia utilities for continuous integration and testing 

* This `Salesforce document <https://developer.salesforce.com/page/How_to_Write_Good_Unit_Tests/>`_ has some good lessons about writing and testing code

Exercises
============

Exercise 1
---------------

Following the `instructions for a new project <project_setup>`_, create a new package on your github account called ``NewtonsMethod.jl``

In this package, you should create a simple package to do Newton's Method using the code you did in the 
`Newton's method <jbe_ex8a>`_ exercise in `Julia by example <julia_by_example>`_

In particular, within your package you should have two functions
* ``newtonroot(f, f′; x₀, tol = 1E-7, maxiter = 1000)``
* ``newtonroot(f; x₀, tol = 1E-7, maxiter = 1000)``

Where the second function one uses Automatic Differentiation to call the first.

The package should include
* implementations of those functions in the ``/src`` directory
* comprehensive set of tests
* project and manifest files to replicate your development environment
* automated running of the tests with Travis CI in GitHub 

For the tests, you should have at the very minimum
* a way to handle non-convergence (e.g. return back ``nothing`` as discussed in `error handling <error_handling>`_
* several ``@test`` for the root of a known function, given the ``f`` and analytical ``f'`` derivatives
* tests of those roots using the automatic differentiation version of the function
* test of finding those roots with a ``BigFloat`` and not just a ``Float64``
* test of non-convergence for a function without a root (e.g. :math:`f(x) = 2 + x^2` )
* test to ensure that the ``maxiter`` is working (e.g. what happens if you call ``maxiter = 5``
* test to ensure that ``tol`` is working

And anything else you can think of.  You should be able to run ``] test`` for the project to check that the test-suite is running, and then ensure that it is running automatically on Travis CI

Push a commit to the repository which breaks one of the tests and see what the Travis CI reports after running the build