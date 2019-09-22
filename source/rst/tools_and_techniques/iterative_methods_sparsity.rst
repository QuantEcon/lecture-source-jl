.. _iterative_methods_sparsity:

.. include:: /_static/includes/header.raw


*****************************************
Iterative Methods and Sparsity
*****************************************

.. contents:: :depth: 2


Overview
============

This lecture takes the structure of :doc:` numerical methods for linear algebra <numerical_linear_algebra>` and builds further


Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics, BenchmarkTools
    seed!(42);  # seed random numbers for reproducibility

Iterative Algorithms for Linear Systems
=======================================

As before, we consider solving the equation 

.. math::

    A x = b

where we will maintain a solution that, if :math:`A` is square, there is a unique solution.  However, we will now
focus on cases where :math:`A` is both massive and sparse (e.g. potentially billions of equations).

While this may seem excessive, it occurs frequently in practice due to the curse of dimensionality and discretizations
of PDEs as well as when working with big or network data.

The methods in the previous section (e.g. factorization and the related guassian elimination) are called direct methods,
and work with matrices that are in-memory, while this section will generalize to linear operators that may or may not be
in memory.

Iterative Methods for Eigensystems
====================================

Frequently you don't need every eigenvalue or eigenvector, and a matrix
may be sufficiently massive that it would be infeasible to use a direct method.