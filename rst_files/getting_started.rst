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

#. Local installation vs. cloud-based solutions

#. Installation of libraries, including the Julia code that underpins these lectures

Jupyter
=========================

Like Python, and unlike Matlab and Stata, there is a looser connection between Julia as a programming language and Julia as a development environment

Because of this, you have much more flexibility in how you write and edit your code, whether
that be locally or on the cloud, in a text-editor or IDE, etc.

One example is `Jupyter <http://jupyter.org/>`_ , which provides 
a browser-based interface to execute code in different languages (e.g. Julia, Python, R, etc.)

While you will eventually use other editors, there are some advantages to starting with Jupyter while learning Julia

* The ability to mix formatted text (including mathematical expressions) and code in a single document

* Nicely formatted output including tables, figures, animation, video, etc.

* Conversion tools to generate PDF slides, static HTML, etc.

* Can be used in the cloud without requiring installation

Whether you end up using Jupyter as your primary work environment or not, you'll find learning about it an excellent investment


.. _jl_jupyterlocal:

Installing Julia and Dependencies
==============================================

.. While using the Docker instance is convenient and error-proof, you may eventually want to install things locally

The easiest approach to using Julia with Jupyter on your desktop is to 
install the latest version of `Anaconda <https://www.anaconda.com/download/#macos>`_ and then Julia

* Install Anaconda by `downloading the binary <https://www.anaconda.com/download/>`_

    * Make sure you click yes to "add Anaconda to my PATH"

* Download and install Julia, from `download page <http://julialang.org/downloads/>`_ , accepting all default options

    * We do not recommend `JuliaPro <https://juliacomputing.com/products/juliapro.html>`_
      due to its limited number of available packages

* Open Julia by navigating to Julia through your menus or desktop icons or 
  typing ``julia`` into a terminal or windows console

Either way you should now be looking at something like this

.. figure:: /_static/figures/julia_term_1.png
   :scale: 100%

This is called the JULIA *REPL* (Read-Evaluate-Print-Loop)

We examine the REPL and its different modes in more detail in the :doc:`tools and editors <tools_editors>` lecture  

We'll start by installing some key Julia packages needed to run code in the lectures

Typing ``]`` into the REPL switches to the package manager, where we can add packages

* In the Julia terminal, type the following

    .. code-block:: julia 
        :class: no-execute

        ] add IJulia InstantiateFromURL; precompile

*Note:* On OS/X you will need to type the ``]`` separately and cannot copy/paste the whole string


.. _jl_jupyter:

Using Jupyter
================


.. _ipython_notebook:

Getting Started
-----------------------

To run Jupyter, open a terminal or windows console, ``cd`` to the location you wish to write files and type 

.. code-block:: none

    jupyter lab

Your web browser should open to a page that looks something like this

.. figure:: /_static/figures/starting_nb_julia.png
   :scale: 100%

The page you are looking at is called the "dashboard"

If you click on "Julia 1.0.x" you should have the option to start a Julia notebook

Here's what your Julia notebook should look like

The notebook displays an *active cell*, which you can type Julia commands into

.. figure:: /_static/figures/nb2_julia.png
   :scale: 100%


.. Not sure this is helpful
.. **Note** The address ``localhost:8888/lab`` you see in the image indicates that the browser is communicating with a Jupyter lab session via port 8888 of the local machine


Modal Editing
^^^^^^^^^^^^^^^^^^^^

The Jupyter notebook uses a *modal* editing system

This means that the effect of typing at the keyboard **depends on which mode you are in**

The two modes are

#. Edit mode (hit ``Enter`` to get edit mode)

    * Indicated by a green border around one cell, as in the pictures above

    * Whatever you type appears as is in that cell

#. Command mode (hit ``Esc`` to get command mode)

    * The green border is replaced by a blue border

    * Key strokes are interpreted as commands --- for example, typing `b` adds a new cell below  the current one

(To learn about other commands available in command mode, go to "Keyboard Shortcuts" in the "Help" menu)

Notice that in the previous figure the cell is surrounded by a blue border

Press ``Enter`` to get edit mode and type the following code into a cell -- hit ``Shift-Enter`` to execute

.. code-block:: julia

    a = 2
    b = 5
    a + b

Working with Files
^^^^^^^^^^^^^^^^^^^^^^^^^

To run an existing Julia file using the notebook you can copy and paste the contents into a cell in the notebook

If it's a long file, however, you have the alternative of

#. Saving the file in your **present working directory**

#. Executing ``include("filename")`` in a cell


The present working directory can be found by executing the command ``pwd()``


Using QuantEcon Lecture Packages
-------------------------------------------

To use the curated set of packages in the QuantEcon lecture notes, 
copy the following text into a notebook cell, and hit ``Shift-Enter`` to run the cell

.. literalinclude:: /_static/includes/deps.jl    

This downloads, installs, and compiles the correct version of all of packages used in the QuantEcon lectures

Depending on your computer, this may take **10-15 minutes** to run the **first-time**, but be virtually instantaneous thereafter

This code can be put at the top of any notebook in order to get a tested set of 
packages compatible with the code in the QuantEcon lectures

More details on packages are explained in a :doc:`later lecture <tools_editors>`



Working with the Notebook
-----------------------------

Let's go over some more Jupyter notebook features --- enough so that we can press ahead with programming


Plots
^^^^^^^

Let's generate some plots

Now try copying the following into a notebook cell and hit ``Shift-Enter``

.. code-block:: julia

    using Plots
    gr(fmt=:png)
    plot(sin, -2π, 2π, label="sin(x)")


**Note**: The "time-to-first-plot" in Julia takes a while, since it needs to precompile everything


Tab Completion
^^^^^^^^^^^^^^^^^^

Tab completion in Jupyter makes it easy to find Julia commands and functions available

For example if you type ``rep`` and hit the tab key you'll get a list of all
commands that start with ``rep``

.. figure:: /_static/figures/nb5_julia.png
   :scale: 80%


.. _gs_help:

Getting Help
^^^^^^^^^^^^^^^

To get help on the Julia function such as ``repmat``, enter ``?repmat``

Documentation should now appear in the browser



Other Content
^^^^^^^^^^^^^^^

In addition to executing code, the Jupyter notebook allows you to embed text, 
equations, figures and even videos in the document

For example, here we enter a mixture of plain text and LaTeX instead of code

.. figure:: /_static/figures/nb6_julia.png
   :scale: 80%

Next we ``Esc`` to enter command mode and then type ``m`` to indicate that we
are writing `Markdown <http://daringfireball.net/projects/markdown/>`_, a mark-up language similar to (but simpler than) LaTeX

(You can also use your mouse to select ``Markdown`` from the ``Code`` drop-down box just below the list of menu items)

Now we ``Shift + Enter`` the Markdown cell to produce this

.. figure:: /_static/figures/nb7_julia.png
   :scale: 80%

   
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


.. _jl_juliaoptions:

Other Ways to Use Jupyter
===================================

There are alternative workflows, such as 

#. Using :ref:`Jupyter on the cloud or a department server <jl_jupyterhub>` (if it is available)
#. Installing the pre-built :ref:`docker-based Julia/Jupyter <jl_jupyterdocker>` from QuantEcon

Eventually, you will want to move from just using Jupyter to using other 
:doc:`tools and editors <tools_editors>` such as `Atom/Juno <http://junolab.org/>`_, but 
don't let the environment get in the way of learning the language

.. _jl_jupyterhub:

Using Jupyter Online
---------------------------

If you have access to a cloud-based solution for Jupyter, then this is typically an easy solution

* Students: ask your department if these resources are available 
* Universities and workgroups: email `contact@quantecon.org <mailto:contact@quantecon.org">`_ for 
  help on setting up a shared JupyterHub instance with precompiled packages ready for these lecture notes
* `JuliaBox <http://www.juliabox.com>`_  tightly controls allowed packages, and **does not** currently support the QuantEcon lectures

.. * JuliaBox (currently having , once it's working. 
..  For example, many Canadian students have access to syzygy.ca
.. * Ask at your university .. 
.. (e.g. `www.syzygy.ca <www.syzygy.ca>`_ and `juliabox.com <www.juliabox.com>`_ )


If you are using an online Jupyter installation for a class, you may not need to do anything to begin using these notebooks

Otherwise, if there are errors when you attempt to use an online JupyterHub, you will need to go open a Jupyter notebook and type

.. code-block:: julia 
    :class: no-execute

    ] add InstantiateFromURL

If this command fails, then your online JupyterHub may not support adding new packages, and 
will not work with the QuantEcon lectures

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
