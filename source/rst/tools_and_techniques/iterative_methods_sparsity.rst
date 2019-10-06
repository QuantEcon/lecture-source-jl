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

Ill-conditioned Matrices
========================

An important consideration in linear algebra, and iterative methods in general is the `condition number <https://en.wikipedia.org/wiki/Condition_number>`_.

Intuitively, the rates of convergence depend on the properties of the spectrum - or alternatively how close to collinear.

Hilbert matrix

Use ``cond``.  The reaosn to use the ``cond`` rather than the determinant is that it is scaleless whereas if you multiple a matrix by 1E-10, it scales the determinannt.




Iterative Methods for Eigensystems
====================================

When you use ``eigen`` on a matrix, it calculates the full spectral decomposition, providing all of the eigenvalues and eigenvectors.

While sometimes this is necessary, a spectral decomposition of a dense, unstructred matrix is one of the costliest :math:`O(N^3)` operations (i.e., it has
one of the largest constants).  For large matrices it is often infeasible. 

Luckily, you frequently only need a few or even a single eigenvector/eigenvalue, which enables a different set of algorithms.

For example, in the case of a discrete time markov chain, to find the stationary distribution we are looking for the
eigenvector associated with the eigenvalue of 1.  As usual, a little linear algebra goes a long way.

From the `Perron-Frobenius theorem <https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem#Stochastic_matrices>`_, the largest eigenvalue of an irreducible stochastic matrix is 1 - the same eigenvalue we are looking for.

Iterative methods for solving eigensystems allow targeting the smallest magnitude, largest magnitude, and many others.  The easiest library
to use is `Arpack.jl <https://julialinearalgebra.github.io/Arpack.jl/latest/>`_.  

.. code-block:: julia

    using Arpack, LinearAlgebra
    N = 1000
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
    A_adjoint = A'

    λ, ϕ = eigs(A_adjoint, nev=1, which=:LM, maxiter=1000)  # Find 1 of the largest magnitude eigenvalue
    ϕ = real(ϕ) ./ sum(real(ϕ))
    λ

Indeed, the ``λ`` is equal to ``1``.   Hint: if you get errors, increase ``maxiter``.

Similarly, for a continuous time Markov Chain, to find the stationary distribution we are looking for the eigenvector associated with ``λ = 0`, which
must be the smallest absolute magnitude.

With our multi-dimensional CTMC in the previous section

.. code-block:: julia

   using SparseArrays
    function markov_chain_product(Q, A)
        M = size(Q, 1)
        N = size(A, 1)
        Q = sparse(Q)
        Qs = blockdiag(fill(Q, N)...)  # create diagonal blocks of every operator
        As = kron(A, sparse(I(M)))
        return As + Qs
    end

    α = 0.1
    N = 4
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))
    A = sparse([-0.1 0.1
        0.2 -0.2])
    M = size(A,1)
    L = markov_chain_product(Q, A)
    L_adjoint = L';

In this case, Arpack.jl does not do well with the singular matrix of a CTMC.  We can use another package
for similar methods called `KrylovKit.jl <https://jutho.github.io/KrylovKit.jl/latest/man/eig/>`_

    using KrylovKit
   using KrylovKit
    λ, ϕ = KrylovKit.eigsolve(L_adjoint, 1, :SR)  # smallest absolute value
    reshape(real(ϕ[end]), N, size(A,1))
    ϕ = real(ϕ) ./ sum(real(ϕ))
    ϕ = reshape(ϕ, N, size(A,1))
    λ

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
