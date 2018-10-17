.. _more_types:

.. include:: /_static/includes/lecture_howto_jl.raw

*********************************
More Julia Types
*********************************

.. contents:: :depth: 2

Overview
============================

In this lecture we give more types

* reshaping arrays
* sparse arrays
* fixed size arrays
* structured arrays

Setup
------------------

Activate the ``QuantEconLecturePackages`` project environment and package versions

.. code-block:: julia 

    using InstantiateFromURL
    activate_github("QuantEcon/QuantEconLecturePackages")
    using LinearAlgebra, Statistics, Compat


Structured Arrays
==================

Here are some functions that create two-dimensional arrays

.. code-block:: julia

    using LinearAlgebra, Statistics

.. code-block:: julia

    diagm(0 => [2, 4])


.. code-block:: julia

    size(diagm(0 => [2, 4]))

Sparse


.. _further_types:

Cover ``mul!``?

