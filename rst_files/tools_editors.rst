.. _tools_editors:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Julia Tools and Editors
******************************************

.. contents:: :depth: 2

while Jupyter based notebooks are an easy way to get started with the language, you will eventually use a variety of tools and editors

Overview
============

Topics:

* Using the REPL
* Using the Package managers
* Setting up and Using Atom


The REPL
======================
While we introduced the REPL in :doc:`earlier <getting_started>` TODO 


First Steps
==============



**Note:** In these lectures we assume you have version 0.6 or later


Assuming there were no problems, you should now be able to start Julia either by

* Navigating to Julia through your menus or desktop icons (Windows, OSX), or

* Opening a terminal and typing ``julia`` (Linux)

Either way you should now be looking at something like this (modulo your operating system --- this is a Linux machine)


.. figure:: /_static/figures/julia_term_1.png
   :scale: 75%



The REPL
-----------

The program that's running here is called the Julia REPL (Read Eval Print Loop) or Julia interpreter

Let's try some basic commands

.. figure:: /_static/figures/julia_term_2.png
   :scale: 75%

The Julia interpreter has the kind of nice features you expect from a modern REPL

For example,

* Pushing the up arrow key retrieves the previously typed command

* If you type ``?`` the prompt will change to ``help?>`` and give you access to online documentation

.. figure:: /_static/figures/julia_term_3.png
   :scale: 75%

You can also type ``;`` to get a shell prompt, at which you can enter shell
commands

.. figure:: /_static/figures/julia_term_4.png
   :scale: 75%

(Here ``ls`` is a UNIX style command that lists directory contents --- your shell commands depend on your operating system)

Below we'll often show interactions with the interpreter as follows

Activate the project environment, ensuring that ``Project.toml`` and ``Manifest.toml`` are in the same location as your notebook

.. code-block:: julia

    using Pkg; Pkg.activate(@__DIR__); #activate environment in the notebook's location

.. code-block:: julia

    x = 10


.. code-block:: julia

    2 * x



Installing Packages
=======================

In these lectures you'll often see statements such as 

.. code-block:: julia

    using Plots

or

.. code-block:: julia

    using QuantEcon

These commands pull in code from some of Julia's `many external Julia code libraries <http://pkg.julialang.org/>`_

For the code to run, you need to install the corresponding package first

Fortunately this is easy using Julia's package management system

For example, let's install `DataFrames <https://github.com/JuliaStats/DataFrames.jl>`_, which provides useful functions and data types for manipulating data sets

.. code-block:: julia

    Pkg.add("DataFrames")

Assuming you have a working Internet connection this should install the DataFrames package

Here's how it looks on our machine (which already has this package installed)


.. figure:: /_static/figures/julia_term_addpkg.png
   :scale: 75%

If you now type ``Pkg.status()`` you'll see ``DataFrames`` and its version number

To pull the functionality from ``DataFrames`` into the current session we type

.. code-block:: julia

    using DataFrames


Now its functions are accessible


.. code-block:: julia

    df = DataFrame(x1=[1, 2], x2=["foo", "bar"])



Keeping your Packages up to Date
-----------------------------------

Running

.. code-block:: julia

    Pkg.update()

will update your installed packages and also update local information on the set of available packages

We **assume throughout** that you keep your packages updated to the latest version!




.. _gs_qe:

QuantEcon 
---------------

`QuantEcon <http://quantecon.org>`_ is an organization that facilitates development of open source code for economic modeling

As well as these lectures, it supports `QuantEcon.jl <http://quantecon.org/julia_index.html>`__, a library for quantitative economic modeling in Julia

The installation method is standard

.. code-block:: julia

    Pkg.add("QuantEcon")


Here's an example, which creates a discrete approximation to an AR(1) process

.. code-block:: julia

    using QuantEcon: tauchen

    tauchen(4, 0.9, 1.0)




We'll learn more about the library as we go along



Alternatives to Jupyter
========================


In this lecture series we'll assume that you're using Jupyter

Doing so allows us to make sure that everything works in at least one sensible environment

But as you work more with Julia you will want to explore other environments as
well

Here are some notes on working with the REPL, text editors and other alternatives


Editing Julia Scripts
-----------------------

You can run Julia scripts from the REPL using the ``include("filename")`` syntax

The file needs to be in the present working directory, which you can determine by typing ``pwd()``


You also need to know how to edit them --- let's discuss how to do this without Jupyter

IDEs
^^^^^

`IDEs <https://en.wikipedia.org/wiki/Integrated_development_environment>`_  (Integrated Development Environments) combine an interpreter and text editing facilities in the one application

For Julia one nice option is `Juno <http://junolab.org/>`_



Text Editors
^^^^^^^^^^^^^

The beauty of text editors is that if you master one of them, you can use it
for every coding task you come across, regardless of the language

At a minimum, a text editor for coding should provide

* Syntax highlighting for the languages you want to work with

* Automatic indentation

* Efficient text manipulation (search and replace, copy and paste, etc.)


There are many text editors that speak Julia, and a lot of them are free

Suggestions:

* `Atom <https://atom.io/>`_ is a popular open source next generation text editor

* `Sublime Text <http://www.sublimetext.com/>`_ is a modern, popular and highly regarded text editor with a relatively moderate learning curve (not free but trial period is unlimited)

* `Emacs <http://www.gnu.org/software/emacs/>`_ is a high quality free editor with a sharper learning curve

Finally, if you want an outstanding free text editor and don't mind a seemingly vertical learning curve plus long days of pain and suffering while all your neural pathways are rewired, try `Vim <http://www.vim.org/>`_



Exercises
===========

Exercise 1
------------

If Jupyter is still running, quit by using ``Ctrl-C`` at the terminal where you started it

Now launch again, but this time using ``jupyter notebook --no-browser``

This should start the kernel without launching the browser

Note also the startup message: It should give you a URL such as ``http://localhost:8888`` where the notebook is running

Now

#. Start your browser --- or open a new tab if it's already running

#. Enter the URL from above (e.g. ``http://localhost:8888``) in the address bar at the top

You should now be able to run a standard Jupyter notebook session

This is an alternative way to start the notebook that can also be handy




Exercise 2
------------

.. index:: 
    single: Git

This exercise will familiarize you with git and GitHub

`Git <http://git-scm.com/>`_ is a *version control system* --- a piece of software used to manage digital projects such as code libraries

In many cases the associated collections of files --- called *repositories* --- are stored on `GitHub <https://github.com/>`_

GitHub is a wonderland of collaborative coding projects

Git is an extremely powerful tool for distributed collaboration --- for
example, we use it to share and synchronize all the source files for these
lectures

There are two main flavors of Git

#. The plain vanilla `command line Git <http://git-scm.com/downloads>`_ version

#. The various point-and-click GUI versions

    * See, for example, the `GitHub version <https://desktop.github.com/>`_

As an exercise, try 

#. Installing Git
   
#. Getting a copy of `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`_ using Git

For example, if you've installed the command line version, open up a terminal and enter

.. code-block:: bash

	git clone https://github.com/QuantEcon/QuantEcon.jl

(This is just ``git clone`` in front of the URL for the repository)

Even better, 

#. Sign up to `GitHub <https://github.com/>`_ 

#. Look into 'forking' GitHub repositories (forking means making your own copy of a GitHub repository, stored on GitHub)

#. Fork `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`_

#. Clone your fork to some local directory, make edits, commit them, and push them back up to your forked GitHub repo

#. If you made a valuable improvement, send us a `pull request <https://help.github.com/articles/about-pull-requests/>`_!

For reading on these and other topics, try

* `The official Git documentation <http://git-scm.com/doc>`_

* Reading through the docs on `GitHub <https://github.com/>`_

* `Pro Git Book <http://git-scm.com/book>`_ by Scott Chacon and Ben Straub

* One of the thousands of Git tutorials on the Net

