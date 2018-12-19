.. _julia_environment:

.. include:: /_static/includes/lecture_howto_jl_full.raw

*********************************************
:index:`Interacting with Julia`
*********************************************

.. contents:: :depth: 2

Overview
==============

In this lecture we'll start examining different features of the Julia and Jupyter environments

Using Jupyter
================

.. _ipython_notebook:

Getting Started
-----------------------

Recall that, to start Jupyter on your local machine, you should ``cd`` there in your terminal and type

.. code-block:: none

   jupyter lab

If you are using an online Jupyter, then you can directly open a new notebook

Your web browser should open to a page that looks something like this

.. figure:: /_static/figures/starting_nb_julia.png
   :scale: 100%

The page you are looking at is called the "dashboard"

If you click on "Julia 1.0.x" you should have the option to start a Julia notebook

Here's what your Julia notebook should look like

.. figure:: /_static/figures/nb2_julia.png
   :scale: 100%

The notebook displays an *active cell*, into which you can type Julia commands

Notebook Basics
------------------

Notice that in the previous figure the cell is surrounded by a blue border

This means that the cell is selected, and double-clicking will place it in edit mode

As a result, you can type in Julia code and it will appear in the cell

When you're ready to execute these commands, hit ``Shift-Enter``

.. figure:: /_static/figures/nb3_julia.png
   :scale: 100%


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

Note that if you're using a JupyterHub setup, you will need to first run

.. literalinclude:: /_static/includes/alldeps_no_using.jl

in a new cell (i.e., ``Shift + Enter``)

This might take 15-20 minutes depending on your setup

Run the following cell

.. code-block:: julia

    using Plots
    gr(fmt=:png);
    plot(sin, -2π, 2π, label="sin(x)")

You'll see something like this (although the style of plot depends on your
installation)

.. figure:: /_static/figures/nb4_julia.png
   :scale: 100%

**Note**: The "time-to-first-plot" in Julia takes a while, since it needs to compile many functions - but is almost instantaneous the second time you run the cell

Working with the Notebook
-----------------------------

Let's go over some more Jupyter notebook features --- enough so that we can press ahead with programming


Tab Completion
^^^^^^^^^^^^^^^^^^

Tab completion in Jupyter makes it easy to find Julia commands and functions available

For example if you type ``rep`` and hit the tab key you'll get a list of all
commands that start with ``rep``

.. figure:: /_static/figures/nb5_julia.png
   :scale: 100%


.. _gs_help:

Getting Help
^^^^^^^^^^^^^^^

To get help on the Julia function such as ``repeat``, enter ``? repeat``

Documentation should now appear in the browser

.. figure:: /_static/figures/repeatexample.png
   :scale: 100%

Other Content
^^^^^^^^^^^^^^^

In addition to executing code, the Jupyter notebook allows you to embed text, equations, figures and even videos in the page

For example, here we enter a mixture of plain text and LaTeX instead of code

.. figure:: /_static/figures/nb6_julia.png
   :scale: 100%

Next we ``Esc`` to enter command mode and then type ``m`` to indicate that we
are writing `Markdown <http://daringfireball.net/projects/markdown/>`_, a mark-up language similar to (but simpler than) LaTeX

(You can also use your mouse to select ``Markdown`` from the ``Code`` drop-down box just below the list of menu items)

Now we ``Shift + Enter`` to produce this

.. figure:: /_static/figures/nb7_julia.png
   :scale: 100%


Inserting unicode (e.g. Greek letters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Julia supports the use of `unicode characters <https://docs.julialang.org/en/v1/manual/unicode-input/>`__
such as ``α`` and ``β`` in your code

Unicode characters can be typed quickly in Jupyter using the ``tab`` key

Try creating a new code cell and typing ``\alpha``, then hitting the ``tab`` key on your keyboard

Shell Commands
^^^^^^^^^^^^^^^^

You can execute shell commands (system commands) in Jupyter by prepending a semicolon

For example, ``; ls`` will execute the UNIX style shell command ``ls``,
which --- at least for UNIX style operating systems --- lists the
contents of the current working directory

These shell commands are handled by your default system shell and hence are platform specific

Package Operations
^^^^^^^^^^^^^^^^^^^

You can execute package operations in the notebook by prepending a ``]``

For example, ``] st`` will give the status of installed packages in the current environment

**Note**: Cells where you use ``;`` and ``]`` must not have any other instructions in them (i.e., they should be one-liners)

Sharing Notebooks
------------------------

Notebook files are just text files structured in `JSON <https://en.wikipedia.org/wiki/JSON>`_ and typically end with ``.ipynb``

A notebook can easily be saved and shared between users --- you just need to
pass around the ``ipynb`` file

To open an existing ``ipynb`` file, import it from the dashboard (the first
browser page that opens when you start Jupyter notebook) and run the cells or edit as discussed above

The Jupyter organization has a site for sharing notebooks called `nbviewer <http://nbviewer.jupyter.org/>`_
which provides a static HTML representations of notebooks

.. Notebook can be downloaded as an ``ipynb`` file by clicking on the download icon at the top right of its page

QuantEcon also hosts the `QuantEcon Notes <http://notes.quantecon.org/>`_ website, where you can upload and share your notebooks with other economists and the QuantEcon community

.. _julia_repl:

Using the REPL
================

As we saw in the :ref:`desktop installation <intro_repl>`, the REPL is a Julia specific terminal

It becomes increasingly important as you learn Julia, and you will find it to be a useful tool for interacting with Julia and installing packages

As a reminder, to open the REPL on your desktop, either

    #. Navigating to Julia through your menus or desktop icons (Windows, Mac), or

    #. Opening a terminal and typing ``julia`` (Linux)

If you are using a JupyterHub installation, you can start the REPL in JupyterLab by choosing

#. Choose "New Launcher"
#. Choose a ``Julia 1.0`` Console

We examine the REPL and its different modes in more detail in the :ref:`tools and editors <repl_main>` lecture
