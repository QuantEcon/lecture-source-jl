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

In particular, we will examine the structure of matrices and linear operators (e.g., dense, sparse, symmetric, tridiagonal, banded) and
discuss how it can be exploited to radically increase the performance of solving the problems.

In :doc:`iterative methods and sparsity <iterative_methods_sparsity>` we take this even 
further, and look at methods for large problems where matrices are generalized as linear operators that don't require literal storage as a matrix.

The list of specialized packages for these tasks is enormous and growing, but some of the important organizations to 
look at are `JuliaMatrices <https://github.com/JuliaMatrices>`_ , `JuilaSparse <https://github.com/JuliaSparse>`_, and `JuliaMath <https://github.com/JuliaMath>`_

**NOTE** This lecture explores techniques linear algebra, with an emphasis on large systems.  It requires more advanced understanding of Julia features (in particular multiple-dispatch and generic programming) so make sure to review :doc:`introduction to types <../getting_starting_julia/introduction_to_types>` carefully, and consider further study on :doc:`generic programming <../more_julia/generic_programming>`.

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics, BenchmarkTools
    seed!(42);  # seed random numbers for reproducibility

Target Problems  
----------------

In this section lecture, we are considering variations on three classic problems.

First is the solution to

.. math::

    A x = b

for a square :math:`A` where we will maintain throughout there is a unique solution. 

On paper, since the `Invertible Matrix Theorem <https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem>`_ tells us a unique solution is
equivalent to :math:`A` being invertible, we often write the solution as

.. math::

    x = A^{-1} b

Second, in the case of a rectangular matrix, :math:`A` we consider the `linear least-squares <https://en.wikipedia.org/wiki/Linear_least_squares>`_ solution
to 

.. math::

    \min_x ||Ax -b||^2

From theory, we know that :math:`A` has linearly independent columns that the solution to this over-determined system
is the `normal equations <https://en.wikipedia.org/wiki/Linear_least_squares#Derivation_of_the_normal_equations>`_

.. math::

    x = (A'A)^{-1}A'b

And finally, consider the eigenvalue problem of finding :math:`x` and :math:`\lambda` such that

.. math::

    A x = \lambda x

For the eigenvalue problems, consider that you do not always require all of the :math:`\lambda`, and sometimes the largest (or smallest) would be enough.

The theme of this lecture, and numerical linear algebra in general, comes down to three priciples:

#. **identify structure** (e.g. symmetry, sparsity, etc.) of :math:`A` in order to use **specialized algorithms**
#. **do not lose structure** by applying the wrong linear algebra operations at the wrong times (e.g. sparse matrix becoming dense)
#. understand the **computational complexity** of each algorithm

Computational Complexity
------------------------

As the goal of this section is to move towards numerical linear algebra of large systems, we need to understand how well algorithms scale with size.  This notion is called `computational complexity <https://en.wikipedia.org/wiki/Computational_complexity>`)_.

While this notion of complexity can work at various levels such as the number of `significant digits <https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Arithmetic_functions>`_ for basic mathematical operations, the amount of memory and storage required, or the amount of time) - but we will typically focus on the time-complexity.

For time-complexity, the ``N`` is usually the dimensionality of the problem, although occasionally the key will be the number of non-zeros in the matrix or width of bands.  For our applications, time-complexity is best thought of as the number of floating point operations (e.g. add, multiply, etc.) required.

Complexity of algorithms is typically written in `Big O <https://en.wikipedia.org/wiki/Big_O_notation>`_ notation which provides bounds on the scaling.

Formally, we can write this as :math:`f(N) = O(g(N)) \text{ as} N \to infty` wher the interpretation is that there exists some constants :math:`M, N_0` such that

.. math::

    f(N) \leq M g(N), \text{ for } N > N_0}

For example, the complexity of finding an LU Decomposition of a dense matrix is :math:`O(N^3)` which should be read as there being a constant where
eventually the number of floating point operations required decompose a matrix of size :math:`N\times N` grows cubically.

Keep in mind that these are asymptoptic results intended to understanding the scaling of the problem, and the constant can matter for a given
fixed size.

For example, the number of operations required for an `LU decomposition <https://en.wikipedia.org/wiki/LU_decomposition#Algorithms>`_ of a dense :math:`N \times N` matrix :math:`2/3 N^3`, ignoring the :math:`N^2` and lower terms.  However, sparse matrix algorithms instead scale with the number of non-zeros in the matrix instead.

Rules of Computational Complexity
------------------------------------

When combining algorithms, you will sometimes need to think through how `combining algorithms  <https://en.wikipedia.org/wiki/Big_O_notation#Properties>`_ changes complexity.  For example, if you do,

#. an :math:`O(N^3)` operation :math:`P` times, then it simply changes the constant and remains :math:`O(N^3)`
#. one :math:`O(N^3)` operation and another :math:`O(N^2)` one, then you take the max, which does not change the scale :math:`O(N^3)`
#. a repetition of a :math:`O(N)` operation that itself uses an :math:`O(N)` one, you take the product, and the complexity becomes :math:`O(N^2)`


With this, there is a word of caution: dense matrix-multiplication is an `expensive operation <https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra>`_ for unstructured matrices, and the basic version is :math:`O(N^3)`.

Of course, modern libraries use highly turned and `careful algorithms <https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm>`_ to multiply matrices and exploit the computer architecture, memory cache, etc., but this simply lowers the constant of proportionality and they remain :math:`O(N^3)`.

Since many algorithms require multiplication, this means that it is usually not possible to go below that order without further matrix structure.

That is, changing the constant of proportionality for a given size helps, but in order to achieve high scaling you need to identify matrix structure and ensure your operations do not lose it.


Losing Structure
----------------

As a first example of a structured matrix, consider a `sparse arrays <https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html>`_

.. math::

    A = sprand(10, 10, 0.45)  # random 10x10, 45% filled with non-zeros

    @show nnz(A)  # counts the number of non-zeros
    invA = sparse(inv(Array(A)))  # Julia won't even invert sparse directly
    @show nnz(invA);

This further demonstrates that significant sparsity can be lost when calculating an inverse

.. math::

    A = sprand(10, 11, 0.5)
    @show nnz(A)
    @show nnz(A'*A);

This can be even more extreme for common matrices, for example consider a tridiagonal matrix of size :math:`N \times N` 
that might come out of a Markov Chain.

.. math::

    N = 5
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])

The number of non-zeros here is approximately :math:`3 N`, linear, which scales well for huge matrices into the millions or billions

But consider the inverse

.. math::

    inv(A)

Now, the matrix is fully dense and scales :math:`N^2`

**Sparsity/Structure is not just for storage**:  While we have been emphasizing counting the non-zeros as a heuristic, the primary reason to maintain structure
and sparsity is not for using less memory to store the matrices. 

Certainly it can sometimes become important (e.g. a 1million by 1 million tridiagonal matrix needs to store 3 million numbers, where a dense one requires 1 trillion)

But, as we will see, the main purpose of considering sparsity and matrix structure is that it enables specialized algoritms which typically
have a lower-computational order than unstrucuted dense, or even an unstructed sparse operations.

.. math::

    N = 1000
    b = rand(N)
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
    A_sparse = sparse(A)
    A_dense = Array(A)

    # solve system A x = b
    @btime($A \ $b)
    @btime($A_sparse \ $b)
    @btime($A_dense \ $b)

This example shows what is at stake:  using a structured tridiagonal is 10x faster than using a sparse matrix which is 100x faster then
using a dense matrix.

Simple Examples
===============
 

First Examples
--------------

To begin, consider a simple linear system of a dense matrix

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)

On paper, we to solve for :math:`A x = b` by inverting the matrix,

.. code-block:: julia

    x = inv(A) * b

As we will see, inverting matrices should be used for theory, not for code.  The classic advice that you should `never invert a matrix <https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix>`_ may be `slightly exaggerated <https://arxiv.org/abs/1201.6035>`_, but is generally good advice.

 
As we will see, solving a system by inverting a matrix is always slower, potentially less accurate, and will lose crucial sparsity




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


Factorizations
===============

.. _jl_decomposition:

LU Decomposition
-------------------

For a general dense matrix without any other structure (i.e. not known to be symmetric, tridiagonal, etc.) the standard approach is to first
factor the matrix iin order to exploit the speed of backward and forward subtitution to complete the solution.

The :math:`LU` decompositions finds a lower triangular :math:`L` and upper triangular :math:`U` such that :math:`L U = A`.

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



Calculating the QR Decomposition
==================================

:ref:`Previously <qr_decomposition>`_, we learned about applications of the QR 


Sparse Matrices
=========================

One of the first types of structured matrices is 

SparseArrays
-----------------------------------

The first is to set up columns and construct a dataframe by assigning names

.. code-block:: julia

    2 == 2

