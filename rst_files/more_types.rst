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

Reshaping
==========


Changing Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^

The primary function for changing the dimension of an array is ``reshape()``


.. code-block:: julia

    a = [10, 20, 30, 40]


.. code-block:: julia

    b = reshape(a, 2, 2)


.. code-block:: julia

    b


Notice that this function returns a "view" on the existing array

This means that changing the data in the new array will modify the data in the
old one:

.. code-block:: julia

    b[1, 1] = 100  # Continuing the previous example


.. code-block:: julia

    b


.. code-block:: julia

    a


To collapse an array along one dimension you can use ``dropdims()``

.. code-block:: julia

    a = [1 2 3 4]  # Two dimensional


.. code-block:: julia

    dropdims(a, dims = 1)


The return value is an array with the specified dimension "flattened"


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


.. _more_types:


