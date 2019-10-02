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

Digression on Allocations and Inplace Operations
================================================

While we have usually not considered optimizing code for performance (and focused on the choice of
algorithms instead), when matrices and vectors become large we need to be more careful.

The most important thing to avoid are excess allocations, which usually occur due to the use of
temporary vectors and matrices when they are not necessary.  However, caution is suggested since
excess allocations are never relevant for scalar values, and can sometimes create faster code for
smaller matrices/vectors.

To see this, a convenient tool is the benchmarking

.. code-block:: julia

    using BenchmarkTools
    function f!(C, A, B)
        D = A*B
        C .= D .+ 1
    end
    @btime f!($C, $A, $B)

The ``!`` on the ``f!`` is an informal way to say that the function is mutating, and the first arguments ``C``
is by convention the modified values.

There, notice that the ``D`` is a temporary variable which is created, and then modified afterwards.  However, notice that since
``C`` is modified directly, there is no need to create the temporary matrix.

This is an example of where an inplace version of the matrix multiplication can help avoid the allocation.

.. code-block:: julia

    function f2!(C, A, B)
        mul!(C, A, B)  # in place multiplication
        C .+= 1
    end
    A = rand(10,10)
    B = rand(10,10)
    C = similar(A)
    @btime f!($C, $A, $B)
    @btime f2!($C, $A, $B)

Note in the output of the benchmarking, the ``f2!`` is non-allocating and is using the preallocated ``C`` variable directly.

Another example of this is solutions to linear equations.

.. code-block:: julia

    A = rand(10,10)
    y = rand(10)
    z = A \ y  # creates temporary

    A = factorize(A)  # inplace requires factorization
    x = similar(y)
    ldiv!(x, A, y)  # inplace left divide, using factorization

However, if you benchmark carefully, you will see that this is sometimes slower.  Avoiding allocations is not always a good
idea.

There are a variety of other non-allocating versions of functions.  For example,

.. code-block:: julia

    A = rand(10,10)
    B = similar(A)

    transpose!(B, A)  # non-allocating version of B = transpose(A)

.. Iterative Algorithms for Linear Systems
.. =======================================

.. As before, we consider solving the equation 

.. .. math::

..     A x = b

.. where we will maintain a solution that, if :math:`A` is square, there is a unique solution.  However, we will now
.. focus on cases where :math:`A` is both massive and sparse (e.g. potentially billions of equations).

.. While this may seem excessive, it occurs frequently in practice due to the curse of dimensionality and discretizations
.. of PDEs as well as when working with big or network data.

.. The methods in the previous section (e.g. factorization and the related guassian elimination) are called direct methods,
.. and work with matrices that are in-memory, while this section will generalize to linear operators that may or may not be
.. in memory.

.. Iterative Methods for Eigensystems
.. ====================================

.. Arpack.  Hint: set the maxiter.

.. Frequently you don't need every eigenvalue or eigenvector, and a matrix
.. may be sufficiently massive that it would be infeasible to use a direct method.

.. ..code-block:: julia

..     using KrylovKit
..     KrylovKit.eigsolve(L, 1, :LR)