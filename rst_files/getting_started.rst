.. _getting_started:

.. include:: /_static/includes/lecture_howto_jl_full.raw

*************************************
Setting up Your Julia Environment
*************************************

.. contents:: :depth: 2

Overview
============

In this lecture we will cover how to get up and running with Julia

Topics:

#. Jupyter

#. Choosing from different options

#. Installation of libraries, including the Julia code that underpins these lectures

Jupyter
=========================

Like Python, and unlike Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a development environment

Because of this, you will have different options for editing code on your local computer or in the cloud

Several of the options rely on `Jupyter <http://jupyter.org/>`_  which provides a browser-based interface to access computational kernels for different languages (e.g. Julia, Python, R, etc.)

While you will eventually use other editors, there are some advantages of starting with Jupyter while learning the language

* It requires no installation if you used a cloud-based solution

* Nicely formatted output in the browser, including tables, figures, animation, video, etc.

* The ability to mix in formatted text and mathematical expressions between cells

* Functions to generate PDF slides, static HTML, etc.

Whether you end up using Jupyter as your primary work environment or not, you'll find learning about it an excellent investment

Options
-----------------------------

There are a few ways to get started with Jupyter

#. Install `Anaconda and Julia locally <jl_jupyterlocal>`_ otherwise and manually install QuantEcon based packages
#. Use `Jupyter on the cloud or a department server <jl_jupyterhub>`_ (if it is available)
#. Install the pre-built `docker-based Julia/Jupyter <jl_jupyterdocker>`_ from QuantEcon

Eventually, you will both want to do a `local installation <jl_jupyterlocal>`_ and move from just using Jupyter to using other `tools and editors <tools_editors>`_ such as `Atom/Juno <http://junolab.org/>`_, but don't let the environment get in the way of learning the language

.. _jl_jupyterhub:

Using Jupyter Online
---------------------------

If you have access to a cloud based solution for Jupyter, then that is typically the easiest solution

* Students: ask your department if these resources are available 
* Universities and workgroups: email `contact@quantecon.org <mailto:contact@quantecon.org">`_ for help on setting up a shared JupyterHub instance with precompiled packages ready for these lecture notes
* `JuliaBox <www.juliabox.com>`_  tightly controls allowed packages, and **does not** currently support the QuantEcon lectures

.. * JuliaBox (currently having , once it's working. 
..  For example, many Canadian students have access to syzygy.ca
.. * Ask at your university .. 
.. (e.g. `www.syzygy.ca <www.syzygy.ca>`_ and `juliabox.com <www.juliabox.com>`_ )


If you are given an online Jupyter installation for a class, you may not need to do anything to begin using these notebooks

Otherwise, if there are errors when you attempt to use an online JupyterHub, you will need to go open a Jupyter notebook and type

.. code-block:: none

    ] add InstantiateFromURL

If this command fails, then your online JupyterHub may not support adding new packages, and will not work with the QuantEcon lecture

.. _jl_jupyterdocker:

Installing a Pre-built Jupyter Image
---------------------------------------

`Docker <https://www.docker.com/>`_ is a technology that you use to host a "`virtual <https://en.wikipedia.org/wiki/Operating-system-level_virtualization>`_ " version of a software setup on another computer

While it is largely used for running code in the cloud, it is also convenient for using on local computers 

QuantEcon has constructed a pre-built `docker image <https://hub.docker.com/u/quantecon/>`_

For instructions on how to set this up, see the `tools and editors <tools_editors>`_ lecture 

**Note:** The Docker installation is easy and complete, but it has limitations on operating systems (in particular, Windows 10 is only supported for the Professional and Education editions, and not the Home edition) 


.. _jl_jupyterlocal:

Installing Julia and Dependencies
==============================================

.. While using the Docker instance is convenient and error-proof, you may eventually want to install things locally

The easiest approach is to using Julia with Jupyter on your desktop is to install Anaconda

1. Install Anaconda by `downloading the binary <https://www.anaconda.com/download/>`_ (3.7 version)

    * Make sure you click yes to "add Anaconda to my PATH."
    * If you would rather do that later, 
        see: `Anaconda for Windows <http://docs.anaconda.com/anaconda/install/windows/>`_ and
        `Mac/Linux <https://conda.io/docs/user-guide/install/macos.html>`_.

.. This could be in a separate section
.. * Note that the packages in Anaconda update regularly --- you can keep up to date by typing ``conda update anaconda`` in a terminal 

2. Download and install Julia, from `download page <http://julialang.org/downloads/>`_ , accepting all default options.

3. Open Julia, either by

    * Navigating to Julia through your menus or desktop icons (Windows, OSX), or
    * Opening a terminal and typing ``julia`` (Linux, + OSX/git bash if you configure it)

Either way you should now be looking at something like this (modulo your operating system)

.. figure:: /_static/figures/julia_term_1.png
   :scale: 75%

4. In that ``julia`` terminal, type the following

    .. code-block:: none

        ] add IJulia InstantiateFromURL Revise REPL

5. Then, install and precompile all of the key packages for these lecture notes (which may take 10-20 minutes)

    .. literalinclude:: /_static/includes/deps.jl    

To run Jupyter, you can now open a terminal, ``cd`` to the location you wish to modify local files in and type 

.. code-block:: none

    jupyter lab

For convenience, you may find it useful on your operating system to change the directory where the REPL starts

6. On Windows, if you have a shortcut on your desktop or on the taskbar, you should: (1) right-click on the icon; (2) right click on the "julia" text; (3) choose "Properties", and (4) change the "Start In" to be something such as ``C:\Users\YOURUSERNAME\Documents``

.. _jl_startup_file:

Creating a Startup File (Advanced)
------------------------------------

Whenever the Julia compiler or REPL starts, it will look for a file called ``startup.jl`` (see `Julia Manual <https://docs.julialang.org/en/v1/manual/getting-started/#man-getting-started-1>`_)

The location for the file is relative to your default Julia environment (e.g. ``~/.julia/config/startup.jl`` or ``C:\Users\USERNAME\.julia\config\startup.jl`` on Windows)

To add one, first create the ``~/.julia/config/`` directory if necessary in the terminal or file explorer  (to find the ``~/.julia`` location in the REPL, you can look at the output of ``] st``)

Next, either download the file `startup.jl </_static/includes/startup.jl>`_ into that directory, or create a file and paste in the following text

.. include:: /_static/includes/startup.jl.raw


.. _jl_jupyter:

Using Jupyter
================


.. _ipython_notebook:

Getting Started
-----------------------

After you have started Jupyter (either on the cloud, the docker, or locally installed on your computer)

You should see something (not exactly) like this

.. figure:: /_static/figures/starting_nb_julia.png
   :scale: 70%

The page you are looking at is called the "dashboard"

The address ``localhost:8888/lab`` you see in the image indicates that the browser is communicating with a Jupyter lab session via port 8888 of the local machine

If you click on "Julia 1.0.1" you should have the option to start a Julia notebook

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

First, ensure that you have activated a set of packages with

.. literalinclude:: /_static/includes/deps.jl    


Now try copying the following into a notebook cell and hit ``Shift-Enter``

.. code-block:: julia

    using Plots
    gr(fmt=:png)
    plot(sin, -2π, 2π, label="sin(x)")

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

   
Inserting unicode (e.g. Greek letters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Julia supports the use of `unicode characters <https://docs.julialang.org/en/v1/manual/unicode-input/>`__
such as α and β in your code

Unicode characters can be typed quickly in Jupyter using the `tab` key

Try creating a new code cell and typing `\\alpha`, then hitting the `tab` key on your keyboard

Shell Commands
^^^^^^^^^^^^^^^^

You can execute shell commands (system commands) in IJulia by prepending a semicolon

For example, ``;ls`` will execute the UNIX style shell command ``ls``, which --- at least for UNIX style operating systems --- lists the contents of the current working directory

These shell commands are handled by your default system shell and hence are platform specific


Package Manager
^^^^^^^^^^^^^^^^

You can enter the package manager by prepending a ``]``

For example, ``] st`` will give the the current status of installed pacakges in the current environment


Sharing Notebooks
------------------------

Notebook files are just text files structured in `JSON <https://en.wikipedia.org/wiki/JSON>`_ and typically end with ``.ipynb``

A notebook can easily be saved and shared between users --- you just need to
pass around the ``ipynb`` file

To open an existing ``ipynb`` file, import it from the dashboard (the first browser page that opens when you start Jupyter notebook) and run the cells or edit as discussed above

The Jupyter organization has a site for sharing notebooks called `nbviewer <http://nbviewer.jupyter.org/>`_ which provides a static HTML representations of notebooks

.. Notebook can be downloaded as an ``ipynb`` file by clicking on the download icon at the top right of its page


The REPL
------------

While we have not emphasized it, on any JupyterHub or locally installed Jupyter installation you will also have access to the Julia REPL

This is a Julia specific terminal disconnected from the graphical interface of Jupyter, and becomes increasingly important as you learn Julia

To start the REPL in a typical Jupyter lab environment

#. Choose "New Launcher"
#. Choose a ``Julia 1.0`` Console

Otherwise, if you  have a local installation, then  

* Navigating to Julia through your menus or desktop icons (Windows, OSX), or

* Opening a terminal and typing ``julia`` (Linux)

The REPL is one of the best places to add and remove packages, so a good test is to see the current status of the package manager

.. code-block:: julia

    ] st
