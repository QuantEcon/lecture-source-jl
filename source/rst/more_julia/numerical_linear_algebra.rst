.. _numerical_linear_algebra:

.. include:: /_static/includes/header.raw


*****************************************
Numerical Methods for Linear Algebra
*****************************************

.. contents:: :depth: 2

.. epigraph::

    You cannot learn too much linear algebra. -- Benedict Gross

Overview
============

This lecture explores some of the key packages for working with linear algebra, with an emphasis on large systems.

In particular, we will examine the structure of matrices and linear operators (e.g., dense, sparse, symmetric, banded) as well
as thinking of matrices in a more general sense as linear operators that don't require literal storage as a matrix.

The list of specialized packages for these tasks is enormous and growing, but some of the important organizations to 
look at are `JuliaMatrices <https://github.com/JuliaMatrices>`_ , `JuilaSparse <https://github.com/JuliaSparse>`_, and `JuliaMath <https://github.com/JuliaMath>`_

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics, BenchmarkTools
    seed!(42);  # seed random numbers for reproducibility

Factorizations
===============

In this section lecture, we are considering variations on the classic 

.. math::

    A x = b

where we will maintain a solution that, if :math:`A` is square, there is a unique solution.  In the case of a rectangular :math:`A`, we will
be looking for a least squares solution of the overdetermined system.

On paper, since the `Invertible Matrix Theorem <https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem>`_ tells us a unique solution is
equivalent to :math:`A` being invertible, we often write the solution as

.. math::

    x = A^{-1} b

And we may even implement this directly in code

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)

    x = inv(A) * b

Nevertheless, inverting matrices should be used for theory, not for code.  The classic advice that you should `never invert a matrix <https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix>`_ may be `slightly exaggerated <https://arxiv.org/abs/1201.6035>`_, but is generally good advice.

 
As we will see, solving a system by inverting a matrix is always slower, potentially less accurate, and will lose crucial sparsity

Instead, use the ``\`` operator, which directly solves the linear system

.. code-block:: julia

    x = A \ b

Behind the scenes, multiple-dispatch is used to determine the best way to solve the equation by choosing the appropriate `factorization <https://en.wikipedia.org/wiki/Matrix_decomposition>`_. 

Triangular Matrices and Back/Forward Substitution
--------------------------------------------------

To begin, consider solving a system with an ``UpperTriangular`` matrix,

.. code-block:: julia

    b = [1.0, 2.0, 3.0]
    U = UpperTriangular([1.0 2.0 3.0; 0.0 5.0 6.0; 0.0 0.0 9.0])

This system is especially easy to solve using `back-substitution <https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution>`_.  In particular, :math:`x_3 = b_3 / U_{33}, x_2 = (b_2 - x_3 * U_{23})/U_{22}`, etc.

.. code-block:: julia

    U \ b


A ``LowerTriangular`` has similar properites and can be solved with forward-substitution.  For these matrices, no further matrix factorization is needed.

LU Decomposition
-------------------

For a general dense matrix without any other structure (i.e. not known to be symmetric, tridiagonal, etc.) one solution approach is to first
factor the matrix into a product of triangular matrices, and then exploit the speed of back and forward subtitution to complete the solution.

This matrix factorization is to find a lower-triangular :math:`L` and upper triangular :math:`U` such that :math:`L U = A`.

We can see which algorithm Julia will use for the ``\`` operator by looking at the ``factorize`` function for a given
matrix.

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)
    
    Af = factorize(A)  # chooses the right factorization, LU here

In this case, it provides an :math:`L` and :math:`U` factorization (with `pivoting <https://en.wikipedia.org/wiki/LU_decomposition#LU_factorization_with_full_pivoting>`_ )

With the factorization complete, we can solve different ``b`` right hand sides

.. code-block:: julia

    b2 = rand(N)
    Af \ b2

To demonstrate what is going on behind the scenes, we can calculate an ``lu`` decomposition without the pivoting, 

.. code-block:: julia

    L, U = lu(A, Val(false))  # the Val(false) provides solution without permutation


And we can verify the decomposition

.. code-block:: julia

    A ≈ L * U

To see roughly how the solver works, note that we can write the problem :math:`A x = b` as :math:`L U x = b`.  Let :math:`U x = y`, which breaks the
problem into two sub-problems.

.. math::

    L y = b
    U x = y

To demonstrate this, we can solve it by first using 

.. code-block:: julia

    y = L \ b

.. code-block:: julia

    x = U \ y


Sparse Matrices
=========================

One of the first types of structured matrices is 

SparseArrays
-----------------------------------

The first is to set up columns and construct a dataframe by assigning names

.. code-block:: julia

    2 == 2

