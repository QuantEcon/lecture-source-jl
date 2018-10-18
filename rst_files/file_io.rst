.. _file_io:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Loading and Storing Files
******************************************

.. contents:: :depth: 2

All source, 

Overview
============

Topics:

* How to manually open and close files
* Working with Panda, R, and other datasets

Setup
------------------

Activate the ``QuantEconLecturePackages`` project environment and package versions

.. code-block:: julia 

    using InstantiateFromURL
    activate_github("QuantEcon/QuantEconLecturePackages")
    using LinearAlgebra, Statistics, Compat

File Input and Output
======================

Let's have a quick look at reading from and writing to text files

We'll start with writing


.. code-block:: julia

    f = open("newfile.txt", "w")  # "w" for writing
    write(f, "testing\n")         # \n for newline
    write(f, "more testing\n")
    close(f)

The effect of this is to create a file called ``newfile.txt`` in your present
working directory with contents


We can read the contents of ``newline.txt`` as follows

.. code-block:: julia

    f = open("newfile.txt", "r")  # Open for reading
    print(read(f, String))
    close(f)

A safer way to read files is the ``d`` notation, which automatically closes the file

.. code-block:: julia

    open("newfile.txt", "r") do f
        print(read(f, String))
    end
