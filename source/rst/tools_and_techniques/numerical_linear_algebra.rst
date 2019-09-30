.. _numerical_linear_algebra:

.. include:: /_static/includes/header.raw


*******************************************
Numerical Linear Algebra and Factorizations
*******************************************

.. contents:: :depth: 2

.. epigraph::

    You cannot learn too much linear algebra. -- Benedict Gross

Overview
============

In particular, we will examine the structure of matrices and linear operators (e.g., dense, sparse, symmetric, tridiagonal, banded) and
discuss how it can be exploited to radically increase the performance of solving the problems.  

We build on :doc:`linear algebra <linear_algebra>`, :doc:`orthogonal projections <orth_proj>`, and :doc:`finite Markov Chains <finite_markov>`.

The methods in this section are called direct methods, and they are qualitatively similar to performing Gaussian elimination to factor matrices and solve systems.  In :doc:`iterative methods and sparsity <iterative_methods_sparsity>` we examine a different approach with iterative algorithms, and generalized the matrices as linear operators.y

The list of specialized packages for these tasks is enormous and growing, but some of the important organizations to
look at are `JuliaMatrices <https://github.com/JuliaMatrices>`_ , `JuliaSparse <https://github.com/JuliaSparse>`_, and `JuliaMath <https://github.com/JuliaMath>`_

**NOTE** This lecture explores techniques linear algebra, with an emphasis on large systems.  You may wish to review multiple-dispatch and generic programming in  :doc:`introduction to types <../getting_starting_julia/introduction_to_types>`, and consider further study on :doc:`generic programming <../more_julia/generic_programming>`.

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics, BenchmarkTools, SparseArrays, Random
    Random.seed!(42);  # seed random numbers for reproducibility

Applications
------------

Some key questions to motivate the lecture.  Is the following a computationally expensive operation as the size of the matrix increases?

- Multiplying two matrices?  It depends.  Multiplying 2 diagonals is trivial.
- Solving a linear system of equations?  It depends.  If the matrix is the identity, the solution is the vector itself.
- Finding the eigenvalues of a matrix?  It depends.  The eigenvalues of a triangular matrix are the diagonal.

With that in mind, in this section lecture, we consider variations on three classic problems.

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

From theory, we know that :math:`A` has linearly independent columns that the solution is the `normal equations <https://en.wikipedia.org/wiki/Linear_least_squares#Derivation_of_the_normal_equations>`_

.. math::

    x = (A'A)^{-1}A'b

And finally, consider the eigenvalue problem of finding :math:`x` and :math:`\lambda` such that

.. math::

    A x = \lambda x

For the eigenvalue problems.  Keep in mind that that you do not always require all of the :math:`\lambda`, and sometimes the largest (or smallest) would be enough.  For example, calculating the spectral radius only requires the maximum eigenvalue in absolute value.

The theme of this lecture, and numerical linear algebra in general, comes down to three principles:

#. **identify structure** (e.g. `symmetric, sparse, diagonal,etc. <https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#Special-matrices-1>`_) of :math:`A` in order to use **specialized algorithms**
#. **do not lose structure** by applying the wrong linear algebra operations at the wrong times (e.g. sparse matrix becoming dense)
#. understand the **computational complexity** of each algorithm

Computational Complexity
------------------------

As the goal of this section is to move towards numerical linear algebra of large systems, we need to understand how well algorithms scale with size.  This notion is called `computational complexity <https://en.wikipedia.org/wiki/Computational_complexity>`_.

While this notion of complexity can work at various levels such as the number of `significant digits <https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Arithmetic_functions>`_ for basic mathematical operations, the amount of memory and storage required, or the amount of time) - but we will typically focus on the time-complexity.

For time-complexity, the ``N`` is usually the dimensionality of the problem, although occasionally the key will be the number of non-zeros in the matrix or width of bands.  For our applications, time-complexity is best thought of as the number of floating point operations (e.g. add, multiply, etc.) required.

Complexity of algorithms is typically written in `Big O <https://en.wikipedia.org/wiki/Big_O_notation>`_ notation which provides bounds on the scaling.

Formally, we can write this as :math:`f(N) = O(g(N)) \text{ as} N \to \infty` wher the interpretation is that there exists some constants :math:`M, N_0` such that

.. math::

    f(N) \leq M g(N), \text{ for } N > N_0

For example, the complexity of finding an LU Decomposition of a dense matrix is :math:`O(N^3)` which should be read as there being a constant where
eventually the number of floating point operations required decompose a matrix of size :math:`N\times N` grows cubically.

Keep in mind that these are asymptotic results intended to understanding the scaling of the problem, and the constant can matter for a given
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

A consequence is that, since many algorithms require matrix-matrix multiplication, it means that it is usually not possible to go below that order without further matrix structure.

That is, changing the constant of proportionality for a given size can help, but in order to achieve higher scaling you need to identify matrix structure and ensure your operations do not lose it.


Losing Structure
----------------

As a first example of a structured matrix, consider a `sparse arrays <https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html>`_

.. code-block:: julia

    A = sprand(10, 10, 0.45)  # random 10x10, 45% filled with non-zeros

    @show nnz(A)  # counts the number of non-zeros
    invA = sparse(inv(Array(A)))  # Julia won't even invert sparse directly
    @show nnz(invA);

This increase from 45 to 100 percent dense demonstrates that significant sparsity can be lost when calculating an inverse.


The results can be even more extreme.  Consider a tridiagonal matrix of size :math:`N \times N`
that might come out of a Markov Chain or discretization  of a diffusion process,

.. code-block:: julia

    N = 5
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])

The number of non-zeros here is approximately :math:`3 N`, linear, which scales well for huge matrices into the millions or billions

But consider the inverse

.. code-block:: julia

    inv(A)

Now, the matrix is fully dense and scales :math:`N^2`

This also applies to the :math:`A' A` operation in the normal equations of LLS.

.. code-block:: julia

    A = sprand(20, 21, 0.3)
    @show nnz(A)/20^2
    @show nnz(A'*A)/21^2;

While there is some variation based on the randoms chosen, we see that a 30 percent dense matrix becomes almost full dense
after the product is taken.

**Sparsity/Structure is not just for storage**:  While we have been emphasizing counting the non-zeros as a heuristic, the primary reason to maintain structure
and sparsity is not for using less memory to store the matrices.

Size can sometimes become important (e.g. a 1 million by 1 million tridiagonal matrix needs to store 3 million numbers (i.e, about 6MB of memory), where a dense one requires 1 trillion (i.e., about 1TB of memory).

But, as we will see, the main purpose of considering sparsity and matrix structure is that it enables specialized algorithms which typically
have a lower-computational order than unstructured dense, or even an unstructured sparse operations.

First, create convenient functions for benchmarking which displays the type

.. code-block:: julia

    using BenchmarkTools
    function benchmark_solve(A, b)
        println("A\\b for typeof(A) = $(string(typeof(A)))")
        @btime $A \ $b
    end  

Then, take away structure to see the impact on performance,

.. code-block:: julia

    N = 1000
    b = rand(N)
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
    A_sparse = sparse(A)
    A_dense = Array(A)

    # benchmark solution to system A x = b and A * B
    benchmark_solve(A, b)
    benchmark_solve(A_sparse, b)
    benchmark_solve(A_dense, b);

This example shows what is at stake:  using a structured tridiagonal may be 10-20x faster than using a sparse matrix which is 100x faster then
using a dense matrix.

In fact, the difference becomes more extreme as the matrices grow.  Solving a tridiagonal system is an :math:`O(N)` while that of a dense matrix without any structure is :math:`O(N^3)`.  The complexity of a sparse solution is more complicated, but roughly scales by :math:`O(nnz(N))`, i.e. the number of nonzeros.


Simple Examples
===============


Inverting Matrices
------------------

To begin, consider a simple linear system of a dense matrix

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)

On paper, we to solve for :math:`A x = b` by inverting the matrix,

.. code-block:: julia

    x = inv(A) * b

As we will see, inverting matrices should be used for theory, not for code.  The classic advice that you should `never invert a matrix <https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix>`_ may be `slightly exaggerated <https://arxiv.org/abs/1201.6035>`_, but is generally good advice.  In fact, the methods used by libraries to invert matrices typically calculate the same factorizations used for computing a system of equations.

To summarize the wisdom: solving a system by inverting a matrix is always a little slower, potentially less accurate, and will often lose crucial sparsity.


Triangular Matrices and Back/Forward Substitution
--------------------------------------------------

To begin, consider solving a system with an ``UpperTriangular`` matrix,

.. code-block:: julia

    b = [1.0, 2.0, 3.0]
    U = UpperTriangular([1.0 2.0 3.0; 0.0 5.0 6.0; 0.0 0.0 9.0])

This system is especially easy to solve using `back-substitution <https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution>`_.  In particular, :math:`x_3 = b_3 / U_{33}, x_2 = (b_2 - x_3 U_{23})/U_{22}`, etc.

.. code-block:: julia

    U \ b


A ``LowerTriangular`` has similar properties and can be solved with forward-substitution.  For these matrices, no further matrix factorization is needed.

The computational order of back-substitution and forward-substitution is :math:`O(N^2)` for dense matrices.

A Warning on Matrix Multiplication
-----------------------------------

Why we write matrix multiplications in our algebra with abandon, in practice the operation scales very poorly without any matrix structure.

Matrix multiplication is so important to modern computers that the constant of scaling in front of the scaling has been radically reduced
when using a proper package, but the order is still :math:`O(N^3)` in practice. 

Sparse matrix multiplication, on the other hand, is :math:`O(N M_A M_B)` where :math:`M_A` are the number of nonzeros per row of :math:`A` and :math:`B` are the number of non-zeros per column of :math:`B`.

By the rules of computational order, that means any algorithm this means that any algorithm requiring a matrix multiplication of dense matrices requires at least :math:`O(N^3)` operation.

The other important question is what is the structure of the resulting matrix. As always, we want to avoid losing 

For example, multiplying an upper triangular by a lower triangular

.. code-block:: julia

    N = 5
    U = UpperTriangular(rand(N,N))

.. code-block:: julia

    L = U'

But the multiplication is fully dense (e.g. think of a cholesky multiplied by itself to produce a covariance matrix)

.. code-block:: julia

    L * U

On the other hand, a tridiagonal times a diagonal is still a tridiagonal and :math:`O(N^2)`

.. code-block:: julia

 A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
 D = Diagonal(rand(N))
 D * A

Factorizations
===============

When you tell a numerical analyst you are solving a linear system directly, their first question is "which factorization?"

.. _jl_decomposition:

LU Decomposition
-------------------

For a general dense matrix without any other structure (i.e. not known to be symmetric, tridiagonal, etc.) the standard approach is to
factor the matrix iin order to exploit the speed of backward and forward substitution to complete the solution.

The computational order of LU decomposition for a dense matrix is :math:`O(N^3)` - the same as Gaussian elimination, but it tends
to have a better constant term than others (e.g. half the number of operations of the QR).  For structured matrices
or sparse ones, that order drops.

The :math:`LU` decompositions finds a lower triangular :math:`L` and upper triangular :math:`U` such that :math:`L U = A`.

We can see which algorithm Julia will use for the ``\`` operator by looking at the ``factorize`` function for a given
matrix.

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)

    Af = factorize(A)  # chooses the right factorization, LU here

In this case, it provides an :math:`L` and :math:`U` factorization (with `pivoting <https://en.wikipedia.org/wiki/LU_decomposition#LU_factorization_with_full_pivoting>`_ ).


With the factorization complete, we can solve different ``b`` right hand sides

.. code-block:: julia

    b2 = rand(N)
    Af \ b2

This decomposition also includes a :math:`P` is a `permutation matrix <https://en.wikipedia.org/wiki/Permutation_matrix>`_ such
that :math:`P A = L U`.

.. code-block:: julia

    Af.P * A ≈ Af.L * Af.U

We can also directly calculate an ``lu`` decomposition without the pivoting,

.. code-block:: julia

    L, U = lu(A, Val(false))  # the Val(false) provides solution without permutation matrices


And we can verify the decomposition

.. code-block:: julia

    A ≈ L * U

To see roughly how the solver works, note that we can write the problem :math:`A x = b` as :math:`L U x = b`.  Let :math:`U x = y`, which breaks the
problem into two sub-problems.

.. math::

    \begin{aligned}
    L y &= b\\
    U x &= y
    \end{aligned}

To demonstrate this, we can solve it by first using

.. code-block:: julia

    y = L \ b

.. code-block:: julia

    x = U \ y
    x ≈ A \ b  # Check identical

The LU decomposition also has specialized algorithms for structured matrices, such as a Tridiagonal

.. code-block:: julia

    N = 1000
    b = rand(N)
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
    factorize(A) |> typeof

This factorization is the key to the performance of the ``A \ b`` in this case.  For Tridiagonal matrices, the
LU decomposition is :math:`O(N^2)`.

Finally, just as a dense matrix without any structure will tend to use a LU decomposition to solve systems,
so will a sparse matrix

.. code-block:: julia

    A_sparse = sparse(A)
    factorize(A_sparse) |> typeof  # dropping the tridiagonal structure to just become sparse

.. code-block:: julia

    benchmark_solve(A, b)
    benchmark_solve(A_sparse, b);

With sparsity, the computational order is related to the number of non-zeros rather than the size of the matrix itself.

Cholesky Decomposition
-----------------------

For real, symmetric, positive definitive matrices, a Cholesky decomposition is a specialized version of the LU decomposition where :math:`L = U'`.


The Cholesky is directly useful on its own (e.g. :doc:`Classical Control with Linear Algebra<../time_series_models/classical_filtering>`) but it is also an efficient factorization to solve symmetric positive definite system.

As always, symmetry allows specialized algorithms.

.. code-block:: julia

    N = 500
    B = rand(N,N)
    A_dense = B' * B  # an easy way to generate a symmetric positive definite matrix
    A = Symmetric(A_dense)

    factorize(A) |> typeof

Here, the :math:`A` decomposition is `Bunch-Kaufman <https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.bunchkaufman>`_ rather than a 
Cholesky, because Julia doesn't know the matrix is positive definite.  We can manually factorize with a cholesky,

.. code-block:: julia

    cholesky(A) |> typeof

Benchmarking,

.. code-block:: julia

    b = rand(N)
    cholesky(A) \ b  # use the factorization to solve

    benchmark_solve(A, b)
    benchmark_solve(A_dense, b)
    @btime cholesky($A, check=false) \ $b;

QR Decomposition
================

:ref:`Previously <qr_decomposition>` , we learned about applications of the QR application to solving the linear least squares.

While in principle, the solution to least-squares is :math:`x = (A'A)^{-1}A'b`, in practice note that :math:`A'A` becomes very dense and inverse are rarely a good idea.  

The QR decomposition is a decomposition :math:`A = Q R` where :math:`Q` is an orthogonal matrix (i.e. :math:`Q'Q = Q Q' = I`) and :math:`R` is
a upper triangular matrix.

Given the  :ref:`previous derivation <qr_decomposition>` we showed that the, given the decomposition, we can write the least squares problem as
the solution to

.. math::

    R x = Q' b

Where, as discussed above, the upper-triangular structure of :math:`R` can be solved easily with back substitution.

For Julia, the ``\`` operator will solve this problem whenever the given ``A`` is rectangular

.. code-block:: julia

    N = 10
    M = 3
    x_true = rand(3)

    A = rand(N,M) .+ randn(N)
    b = rand(N)
    x = A \ b

To manually use the QR decomposition: **Note** the real code would be more subtle

.. code-block:: julia

    Af = qr(A)
    Q = Af.Q
    R = [Af.R; zeros(N - M, M)] # Stack with zeros
    @show Q * R ≈ A
    x = R \ Q'*b  # the QR way

This stacks the ``R`` with zeros to multiple, but the more specialized algorithm would not multiply directly
in that way.

In some cases, if an LU is not available for a particular matrix structure, the QR factorization
can also be used to solve systems of equations (i.e. not just LLS).  This tends to be about 2x slower than the LU,
but is of the same computational order.

Deriving the approach, where we can now use inverse since the system is square and we assumed :math:`A` was non-singular

.. math::

    \begin{aligned}
    A x &= b\\
    Q R x &= b\\
    Q^{-1} Q R x &= Q^{-1} b\\
    R x &= Q' b
    \end{aligned}

Where the last step uses that :math:`Q^{-1} = Q'` for orthogonal matrix.

Given the decomposition, the solution for dense matrices is of computational
order :math:`O(N^2)`.  To see this, look at the order of each operation.

- Since :math:`R` is upper-triangular matrix, it can be solved quickly through back substitution with computational order :math:`O(N^2)`
- A transpose operation is of order :math:`O(N^2)`
- A matrix-vector product is also :math:`O(N^2)`

In all cases, the order would drop depending on the sparsity pattern of the
matrix (and corresponding decomposition).  A key benefit of a QR decomposition is that it tends to
maintain sparsity.

Without implementing the full process, you can form a QR
factorization with ``qr`` and then use it to solve a system

.. code-block:: julia

    N = 5
    A = rand(N,N)
    b = rand(N)
    @show A \ b
    @show qr(A) \ b;   

Banded Matrices
===============

A tridiagonal matrix has 3 non-zero diagonals.  The main diagonal, the first sub-diagonal (i.e. below the main diagonal) and the also the first super-diagonal (i.e. above the main diagonal).

This is a special case of a more general type called a banded matrix, where the number of sub and super-diagonals are more general.  The 
total width of sub- and super-diagonals is called the bandwidth.  For example, a tridiagonal matrix has a bandwidth of 3.

A :math:`N \times N` banded matrix with bandwidth :math:`P` has about :math:`N P` nonzeros in its sparsity pattern.

These can be created directly as a dense matrix with ``diagm``

.. code-block:: julia

    diagm(1 => [1,2,3], -1 => [4,5])

Or as a sparse matrix,

.. code-block:: julia

    spdiagm(1 => [1,2,3], -1 => [4,5])

Creating a simple banded matrix, using `BandedMatrices.jl <https://github.com/JuliaMatrices/BandedMatrices.jl>`_ 

.. code-block:: julia

    using BandedMatrices
    BandedMatrix(-1=> 1:5, 2=>1:3)     # creates a 5 x 5 banded matrix version of diagm(-1=> 1:5, 2=>1:3)

There is also a convenience function for generating random banded matrices

.. code-block:: julia

    brand(7, 7, 3, 1)  # 3 subdiagonals, 1 subdiagonal

And, of course, specialized algorithms will be used to exploit the structure when solving linear systems.  In particular, the complexity is related to the :math:`O(N P_L P_U)` for upper and lower bandwidths :math:`P`

.. code-block:: julia

    A \ rand(7)

    
Continuous Time Markov Chains
=============================

In the previous lecture on :doc:`discrete time Markov Chains  <mc>`, we saw that the transition probability
between state :math:`x` and state :math:`y` was summarized by the matrix :math:`P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}`.

As a brief introduction to continuous time processes, consider same state-space as in the discrete
case: :math:`S` a finite set with :math:`n` elements :math:`\{x_1, \ldots, x_n\}`.

A **Markov chain** :math:`\{X_t\}` on :math:`S` is a sequence of random variables on :math:`S` that have the **Markov property**

In continuous time, the `Markov Property <https://en.wikipedia.org/wiki/Markov_property>`_ is more complicated, but intuitively is
the same as the discrete time case.  That is, knowing the current state is enough to know probabilities for future states.

Heuristically, consider a time period :math:`t` and a small step forward :math:`\Delta`.  Then the probability to transition from state :math:`i` to
state :math:`j` is

.. math::

    \mathbb P \{ X(t + \Delta) = j  \,|\, X(t) \} = \begin{cases} q_{ij} \Delta + o(\Delta) & i \neq j\\
                                                                  1 + q_{ii} \Delta + o(\Delta) & i = j \end{cases}

where :math:`q_{ij}` are parameters governing the transition process, and :math:`o(\Delta)` is `little-o notation <https://en.wikipedia.org/wiki/Big_O_notation#Little-o_notation>`_,.  That is, :math:`\lim_{\Delta\to 0} o(\Delta)/\Delta = 0`.

Just as in the discrete case, we can summarize these parameters by a :math:`N \times N` matrix, :math:`Q \in R^{N\times N}`.

The :math:`Q` matrix is called the intensity matrix, or the infinitesimal generator of the Markov Chain.  For example,

.. math::

    Q = \begin{bmatrix} -0.1 & 0.1  & 0 & 0 & 0 & 0\\
                        0.1  &-0.2  & 0.1 &  0 & 0 & 0\\
                        0 & 0.1 & -0.2 & 0.1 & 0 & 0\\
                        0 & 0 & 0.1 & -0.2 & 0.1 & 0\\
                        0 & 0 & 0 & 0.1 & -0.2 & 0.1\\
                        0 & 0 & 0 & 0 & 0.1 & -0.1\\
        \end{bmatrix}

In that example, transitions only occur between adjacent states with the same intensity (except for a ``bouncing'' back of the bottom and top states)

This also demonstrates that the elements of the intensity matrix are not probabilities.  Unlike the discrete case, where every row must sum to one, the rows of :math:`Q` sum to zero, where the diagonal is the only negative values.  that is

- :math:`q_{ij} \geq 0` for :math:`i \neq j`
- :math:`q_{ii} \leq 0`
- :math:`\sum_{j} q_{ij} = 0`

Implementing this :math:`Q` in code

.. code-block:: julia

    using LinearAlgebra
    α = 0.1
    N = 6
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))

Here we use a Tridiagonal to exploit the structure of the problem.

Consider a simple payoff vector :math:`p` associated with each state, and a discount rate :math:`ρ`.  Then we can solve for
the expected present discounted value in a similar way to the discrete time case.

.. math::

    \rho v = p + Q v 

or rearranging slightly, solving the linear system

.. math::

    (\rho I - Q) v = p

For our example, exploiting the tr

.. code-block:: julia

    p = range(0.0, 10.0, length=N)
    ρ = 0.05

    A = ρ * I - Q

Note that this :math:`A` matrix is maintaining the tridiagonal structure of the problem, which leads to an efficient solution to the
linear problem.

.. code-block:: julia

    v = A \ p


The :math:`Q` is also used to calculate the evolution of the Markov chain, in direct analogy to the :math:`ψ_{t+k} = ψ_t P^k` evolution with transition matrix :math:`P` of the discrete case.

In the continuous case, this becomes the system of linear differential equations

.. math::

    \dot{ψ}(t) = Q(t)^T ψ(t) 

given the initial condition :math:`ψ(0)` and where the :math:`Q(t)` intensity matrix is allows to vary with time.  In the simplest case of a constant :math:`Q` matrix, this is a simple constant-coefficient system of Linear ODEs with coefficients :math:`Q^T`

If a stationary equilibria exists, note that :math:`\dot{ψ}(t) = 0`, and the stationary solution :math:`ψ^{*}` would need to fulfill


.. math::

    0 = Q^T ψ^{*}


Notice that this is of the form :math:`0 ψ^{*} = Q^T ψ^{*}` and hence is equivalent to finding the eigevector associated with the :math:`\lambda = 0` eigenvalue

With our example, we can calculate all of the eigenvalues and eigenvectors

.. code-block:: julia

    λ, vecs = eigen(Array(Q'))

Indeed, there is a :math:`\lambda = 0` eigenvalue, which is associated with the last column in the eigenvector.  To turn that into a probability
we need to normalize it.

.. code-block:: julia

    vecs[:,N] ./ sum(vecs[:,N])
