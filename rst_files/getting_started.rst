.. _getting_started:

.. include:: /_static/includes/lecture_howto_jl.raw

*************************************
Setting up Your Julia Environment
*************************************

.. contents:: :depth: 2

Overview
============

In this lecture we will cover how to get up and running with Julia

Topics:

#. Installation

#. Interactive Julia sessions

#. Running sample programs

#. Installation of libraries, including the Julia code that underpins these lectures




First Steps
==============



Installation
---------------

To install Julia, get the current release from the `download page <http://julialang.org/downloads/>`_

**Note:** In these lectures we assume you have version 0.6 or later

Unless you have good reason to do otherwise, choose 

* The current release rather than nightly build 
  
* The platform specific binary rather than source

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


.. _jl_jupyter:

Jupyter
================

To work with Julia in a scientific context we need at a minimum

#. An environment for editing and running Julia code

#. The ability to generate figures and graphics

One option that provides these features is `Jupyter <http://jupyter.org/>`_

As a bonus, Jupyter also provides

* Nicely formatted output in the browser, including tables, figures, animation, video, etc.

* The ability to mix in formatted text and mathematical expressions between cells

* Functions to generate PDF slides, static HTML, etc.

Whether you end up using Jupyter as your primary work environment or not, you'll find learning about it an excellent investment


Installing Jupyter 
------------------------

There are two steps here:

#. Installing Jupyter itself

#. Installing `IJulia <https://github.com/JuliaLang/IJulia.jl>`_, which serves as an interface between Jupyter notebooks and Julia

In fact you can get both by installing IJulia

**However**, if you have the bandwidth, we recommend that you 

#. Do the two steps separately

#. In the first step, when installing Jupyter, do this by installing the larger package `Anaconda Python <https://www.anaconda.com/what-is-anaconda/>`_

The advantage of this approach is that Anaconda gives you not just Jupyter but the whole scientific Python ecosystem

This includes things like plotting tools we'll make use of later



Installing Anaconda
^^^^^^^^^^^^^^^^^^^^^^^

.. _install_anaconda:

Installing Anaconda is straightforward: `download the binary <https://www.anaconda.com/download/>`_ and follow the instructions

If you are asked during the installation process whether you'd like to make Anaconda your default Python installation, say yes --- you can always remove it later

Otherwise you can accept all of the defaults

Note that the packages in Anaconda update regularly --- you can keep up to date by typing ``conda update anaconda`` in a terminal


Installing IJulia
^^^^^^^^^^^^^^^^^^^^^

Now open up a Julia terminal and type

.. code-block:: julia

    Pkg.add("IJulia")

If you have problems, consult `the installation instructions <https://github.com/JuliaLang/IJulia.jl#installation>`_




Other Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^


Since IJulia runs in the browser it might be a good time to update your browser

One good option is to install a free modern browser such as `Chrome <https://www.google.com/chrome/browser/>`_ or `Firefox <https://www.mozilla.org/en-US/firefox/new/>`_

In our experience Chrome plays well with IJulia




.. _ipython_notebook:

Getting Started
-----------------------

Now either 

#. search for and start the Jupyter notebook application on your machine or
   
#. open up a terminal (or `cmd` in Windows) and type ``jupyter notebook``


You should see something (not exactly) like this

.. figure:: /_static/figures/starting_nb_julia.png
   :scale: 70%

The page you are looking at is called the "dashboard"

The address ``localhost:8888/tree`` you see in the image indicates that the browser is communicating with a Julia session via port 8888 of the local machine

If you click on "New" you should have the option to start a Julia notebook

.. figure:: /_static/figures/starting_nb_julia_options.png
   :scale: 70%

Here's what your Julia notebook should look like

.. figure:: /_static/figures/nb2_julia.png
   :scale: 70%

The notebook displays an *active cell*, into which you can type Julia commands



Notebook Basics
------------------

Notice that in the previous figure the cell is surrounded by a green border

This means that the cell is in *edit mode*

As a result, you can type in Julia code and it will appear in the cell

When you're ready to execute these commands, hit ``Shift-Enter`` instead of the usual ``Enter``

.. figure:: /_static/figures/nb3_julia.png
   :scale: 70%


Modal Editing
^^^^^^^^^^^^^^^^^^^^

The next thing to understand about the Jupyter notebook is that it uses a *modal* editing system

This means that the effect of typing at the keyboard **depends on which mode you are in**

The two modes are

#. Edit mode

    * Indicated by a green border around one cell, as in the pictures above

    * Whatever you type appears as is in that cell

#. Command mode

    * The green border is replaced by a blue border

    * Key strokes are interpreted as commands --- for example, typing `b` adds a new cell below  the current one


(To learn about other commands available in command mode, go to "Keyboard Shortcuts" in the "Help" menu)


Switching modes
^^^^^^^^^^^^^^^^^

* To switch to command mode from edit mode, hit the ``Esc`` key

* To switch to edit mode from command mode, hit ``Enter`` or click in a cell

The modal behavior of the Jupyter notebook is a little tricky at first but very efficient when you get used to it



Working with Files
^^^^^^^^^^^^^^^^^^^^^^^^^

To run an existing Julia file using the notebook you can copy and paste the contents into a cell in the notebook

If it's a long file, however, you have the alternative of

#. Saving the file in your **present working directory**

#. Executing ``include("filename")`` in a cell


The present working directory can be found by executing the command ``pwd()``





Plots
^^^^^^^

Let's generate some plots

There are several options we'll :doc:`discuss in detail later <julia_plots>` 

For now lets start with ``Plots.jl``

.. code-block:: julia

    Pkg.add("Plots")

Now try copying the following into a notebook cell and hit ``Shift-Enter``

.. code-block:: julia

    using Plots
    plot(sin, -2pi, pi, label="sine function")


You'll see something like this (although the style of plot depends on your
installation --- more on this later)

.. figure:: /_static/figures/nb4_julia.png
   :scale: 70%


Working with the Notebook
-----------------------------

Let's go over some more Jupyter notebook features --- enough so that we can press ahead with programming



Tab Completion
^^^^^^^^^^^^^^^^^^

A simple but useful feature of IJulia is tab completion

For example if you type ``rep`` and hit the tab key you'll get a list of all
commands that start with ``rep``

.. figure:: /_static/figures/nb5_julia.png
   :scale: 70%

IJulia offers up the possible completions

This helps remind you of what's available and saves a bit of typing


.. _gs_help:

Online Help
^^^^^^^^^^^^^^^

To get help on the Julia function such as ``repmat``, enter ``?repmat``

Documentation should now appear in the browser



Other Content
^^^^^^^^^^^^^^^

In addition to executing code, the Jupyter notebook allows you to embed text, equations, figures and even videos in the page

For example, here we enter a mixture of plain text and LaTeX instead of code

.. figure:: /_static/figures/nb6_julia.png
   :scale: 70%

Next we ``Esc`` to enter command mode and then type ``m`` to indicate that we
are writing `Markdown <http://daringfireball.net/projects/markdown/>`_, a mark-up language similar to (but simpler than) LaTeX

(You can also use your mouse to select ``Markdown`` from the ``Code`` drop-down box just below the list of menu items)

Now we ``Shift + Enter`` to produce this

.. figure:: /_static/figures/nb7_julia.png
   :scale: 70%

   
Inserting unicode (e.g., Greek letters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Julia supports the use of `unicode characters <https://docs.julialang.org/en/release-0.4/manual/unicode-input/>`__
such as α and β in your code

Unicode characters can be typed quickly in Jupyter using the `tab` key

Try creating a new code cell and typing `\\alpha`, then hitting the `tab` key on your keyboard

Shell Commands
^^^^^^^^^^^^^^^^

You can execute shell commands (system commands) in IJulia by prepending a semicolon

For example, ``;ls`` will execute the UNIX style shell command ``ls``, which --- at least for UNIX style operating systems --- lists the contents of the current working directory

These shell commands are handled by your default system shell and hence are platform specific




Sharing Notebooks
------------------------


Notebook files are just text files structured in `JSON <https://en.wikipedia.org/wiki/JSON>`_ and typically end with ``.ipynb``

A notebook can easily be saved and shared between users --- you just need to
pass around the ``ipynb`` file

To open an existing ``ipynb`` file, import it from the dashboard (the first browser page that opens when you start Jupyter notebook) and run the cells or edit as discussed above

nbviewer
^^^^^^^^^^^

The Jupyter organization has a site for sharing notebooks called `nbviewer <http://nbviewer.jupyter.org/>`_

The notebooks you see there are static HTML representations of notebooks

However, each notebook can be downloaded as an ``ipynb`` file by clicking on the download icon at the top right of its page

Once downloaded you can open it as a notebook, as we discussed just above



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

