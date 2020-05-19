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

In this lecture, we examine the structure of matrices and linear operators (e.g., dense, sparse, symmetric, tridiagonal, banded) and
discuss how the structure can be exploited to radically increase the performance of solving large problems.

We build on applications discussed in previous lectures: :doc:`linear algebra <linear_algebra>`, :doc:`orthogonal projections <orth_proj>`, and :doc:`Markov chains <finite_markov>`.

The methods in this section are called direct methods, and they are qualitatively similar to performing Gaussian elimination to factor matrices and solve systems of equations.  In :doc:`iterative methods and sparsity <iterative_methods_sparsity>` we examine a different approach, using iterative algorithms, where we can think of more general linear operators.

The list of specialized packages for these tasks is enormous and growing, but some of the important organizations to
look at are `JuliaMatrices <https://github.com/JuliaMatrices>`_ , `JuliaSparse <https://github.com/JuliaSparse>`_, and `JuliaMath <https://github.com/JuliaMath>`_

*NOTE*: As this section uses advanced Julia techniques, you may wish to review multiple-dispatch and generic programming in  :doc:`introduction to types <../getting_starting_julia/introduction_to_types>`, and consider further study on :doc:`generic programming <../more_julia/generic_programming>`.

The theme of this lecture, and numerical linear algebra in general, comes down to three principles:

#. **Identify structure** (e.g. `symmetric, sparse, diagonal, etc. <https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#Special-matrices-1>`_) of matrices in order to use **specialized algorithms.**
#. **Do not lose structure** by applying the wrong numerical linear algebra operations at the wrong times (e.g. sparse matrix becoming dense)
#. Understand the **computational complexity** of each algorithm, given the structure of the inputs.

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics, BenchmarkTools, SparseArrays, Random
    Random.seed!(42);  # seed random numbers for reproducibility



Computational Complexity
------------------------

Ask yourself whether the following is a **computationally expensive** operation as the matrix **size increases**

- Multiplying two matrices?

    - *Answer*: It depends.  Multiplying 2 diagonal matrices is trivial.

- Solving a linear system of equations?

    - *Answer*: It depends.  If the matrix is the identity, the solution is the vector itself.

- Finding the eigenvalues of a matrix?

    - *Answer*: It depends.  The eigenvalues of a triangular matrix are the diagonal.

As the goal of this section is to move towards numerical methods with large systems, we need to understand how well algorithms scale with the size of matrices/vectors/etc.  This is known as `computational complexity <https://en.wikipedia.org/wiki/Computational_complexity>`_.  As we saw in the answer to the questions above, the algorithm - and hence the computational complexity - changes based on matrix structure.

While this notion of complexity can work at various levels such as the number of `significant digits <https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Arithmetic_functions>`_ for basic mathematical operations, the amount of memory and storage required, or the amount of time - we will typically focus on the time-complexity.

For time-complexity, the size :math:`N` is usually the dimensionality of the problem, although occasionally the key will be the number of non-zeros in the matrix or width of bands.  For our applications, time-complexity is best thought of as the number of floating point operations (e.g. addition, multiplication, etc.) required.

Notation
~~~~~~~~

Complexity of algorithms is typically written in `Big O <https://en.wikipedia.org/wiki/Big_O_notation>`_ notation which provides bounds on the scaling of the computational complexity with respect to the size of the inputs.

Formally, if the number of operations required for a problem size :math:`N` is :math:`f(N)`, we  can write this as :math:`f(N) = O(g(N))` for some :math:`g(N)` - typically a polynomial.

The interpretation is that there exists some constants :math:`M` and :math:`N_0` such that

.. math::

    f(N) \leq M g(N), \text{ for } N > N_0

For example, the complexity of finding an LU Decomposition of a dense matrix is :math:`O(N^3)` which should be read as there being a constant where
eventually the number of floating point operations required to decompose a matrix of size :math:`N\times N` grows cubically.

Keep in mind that these are asymptotic results intended for understanding the scaling of the problem, and the constant can matter for a given
fixed size.

For example, the number of operations required for an `LU decomposition <https://en.wikipedia.org/wiki/LU_decomposition#Algorithms>`_ of a dense :math:`N \times N` matrix is :math:`f(N) = \frac{2}{3} N^3`, ignoring the :math:`N^2` and lower terms.  Other methods of solving a linear system may have different constants of proportionality, even if they have the same scaling :math:`O(N^3)`.

Rules of Computational Complexity
------------------------------------

You will sometimes need to think through how `combining algorithms  <https://en.wikipedia.org/wiki/Big_O_notation#Properties>`_ changes complexity.  For example, if you use

#. An :math:`O(N^3)` operation :math:`P` times, then it simply changes the constant. The complexity remains :math:`O(N^3)`.
#. One :math:`O(N^3)` operation and another :math:`O(N^2)` one, then you take the max.  The complexity remains :math:`O(N^3)`.
#. A repetition of a :math:`O(N)` operation that itself uses an :math:`O(N)` one, you take the product.  The complexity becomes :math:`O(N^2)`.


With this, we have an important word of caution: dense matrix-multiplication is an `expensive operation <https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra>`_ for unstructured matrices.  The naive version is :math:`O(N^3)` while the fastest known algorithms (e.g Coppersmith-Winograd) are roughly :math:`O(N^{2.37})`.  In practice, it is reasonable to crudely approximate with :math:`O(N^3)` when doing an analysis, in part since the higher constant factors of the better scaling algorithms dominate the better complexity until matrices become very large.

Of course, modern libraries use highly tuned and numerically stable `algorithms <https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm>`_ to multiply matrices and exploit the computer architecture, memory cache, etc., but this simply lowers the constant of proportionality and they remain roughly approximated by  :math:`O(N^3)`.

A consequence is that, since many algorithms require matrix-matrix multiplication, it is often not possible to go below that order without further matrix structure.

That is, changing the constant of proportionality for a given size can help, but in order to achieve better scaling you need to identify matrix structure (e.g. tridigonal, sparse, etc.) and ensure your operations do not lose it.


Losing Structure
----------------

As a first example of a structured matrix, consider a `sparse arrays <https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html>`_.

.. code-block:: julia

    A = sprand(10, 10, 0.45)  # random sparse 10x10, 45 percent filled with non-zeros

    @show nnz(A)  # counts the number of non-zeros
    invA = sparse(inv(Array(A)))  # julia won't invert sparse so convert to dense with Array.
    @show nnz(invA);

This increase from less than 50 to 100 percent dense demonstrates that significant sparsity can be lost when computing an inverse.


The results can be even more extreme.  Consider a tridiagonal matrix of size :math:`N \times N`
that might come out of a Markov Chain or a discretization of a diffusion process,

.. code-block:: julia

    N = 5
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])

The number of non-zeros here is approximately :math:`3 N`, linear, which scales well for huge matrices into the millions or billions

But consider the inverse

.. code-block:: julia

    inv(A)

Now, the matrix is fully dense and has :math:`N^2` non-zeros.

This also applies to the :math:`A' A` operation when forming the normal equations of linear-least squares.

.. code-block:: julia

    A = sprand(20, 21, 0.3)
    @show nnz(A)/20^2
    @show nnz(A'*A)/21^2;

We see that a 30 percent dense matrix becomes almost full dense after the product is taken.

*Sparsity/Structure is not just for storage*:  Matrix size can sometimes become important (e.g. a 1 million by 1 million tridiagonal matrix needs to store 3 million numbers (i.e, about 6MB of memory), where a dense one requires 1 trillion (i.e., about 1TB of memory).

But, as we will see, the main purpose of considering sparsity and matrix structure is that it enables specialized algorithms which typically
have a lower-computational order than unstructured dense, or even unstructured sparse operations.

First, create a convenient functions for benchmarking linear solvers

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
    A_sparse = sparse(A)  # sparse but losing tridiagonal structure
    A_dense = Array(A)    # dropping the sparsity structure, dense 1000x1000

    # benchmark solution to system A x = b
    benchmark_solve(A, b)
    benchmark_solve(A_sparse, b)
    benchmark_solve(A_dense, b);

This example shows what is at stake:  using a structured tridiagonal matrix may be 10-20x faster than using a sparse matrix which is 100x faster than
using a dense matrix.

In fact, the difference becomes more extreme as the matrices grow.  Solving a tridiagonal system is :math:`O(N)` while that of a dense matrix without any structure is :math:`O(N^3)`.  The complexity of a sparse solution is more complicated, and scales in part by the ``nnz(N)``, i.e. the number of nonzeros.

Matrix Multiplication
-----------------------------------

While we write matrix multiplications in our algebra with abundance, in practice the computational operation scales very poorly without any matrix structure.

Matrix multiplication is so important to modern computers that the constant of scaling is small using proper packages, but the order is still roughly :math:`O(N^3)` in practice (although smaller in theory, as discussed above).

Sparse matrix multiplication, on the other hand, is :math:`O(N M_A M_B)` where :math:`M_A` are the number of nonzeros per row of :math:`A` and :math:`M_B` are the number of non-zeros per column of :math:`B`.

By the rules of computational order, that means any algorithm requiring a matrix multiplication of dense matrices requires at least :math:`O(N^3)` operation.

The other important question is what is the structure of the resulting matrix.  For example, multiplying an upper triangular by a lower triangular

.. code-block:: julia

    N = 5
    U = UpperTriangular(rand(N,N))

.. code-block:: julia

    L = U'

But the product is fully dense (e.g. think of a cholesky multiplied by itself to produce a covariance matrix)

.. code-block:: julia

    L * U

On the other hand, a tridiagonal times a diagonal is still a tridiagonal - and can use specialized :math:`O(N)` algorithms.

.. code-block:: julia

 A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
 D = Diagonal(rand(N))
 D * A

Factorizations
==============

When you tell a numerical analyst you are solving a linear system using direct methods, their first question is "which factorization?".

Just as you can factorize a number (e.g. :math:`6 = 3 \times 2`) you can factorize a matrix as the product of other, more
convenient matrices (e.g. :math:`A = L U` or :math:`A = Q R` where :math:`L, U, Q,` and :math:`R` have properties such as being triangular or `orthogonal <https://en.wikipedia.org/wiki/Orthogonal_matrix>`_, etc.).

Inverting Matrices
------------------

On paper, since the `Invertible Matrix Theorem <https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem>`_ tells us a unique solution is
equivalent to :math:`A` being invertible, we often write the solution to :math:`A x = b` as

.. math::

    x = A^{-1} b

What if we do not (directly) use a factorization?

Take a simple linear system of a dense matrix,

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)

On paper, we try to solve the system :math:`A x = b` by inverting the matrix,

.. code-block:: julia

    x = inv(A) * b

As we will see throughout, inverting matrices should be used for theory, not for code.  The classic advice that you should `never invert a matrix <https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix>`_ may be `slightly exaggerated <https://arxiv.org/abs/1201.6035>`_, but is generally good advice.

Solving a system by inverting a matrix is always a little slower, potentially less accurate, and will sometimes lose crucial sparsity compared to using factorizations.  Moreover, the methods used by libraries to invert matrices are frequently the same factorizations used for computing a system of equations.

Even if you need to solve a system with the same matrix multiple times, you are better off factoring the matrix and using the solver rather than calculating an inverse.

.. code-block:: julia

    N = 100
    A = rand(N,N)
    M = 30
    B = rand(N,M)
    function solve_inverting(A, B)
        A_inv = inv(A)
        X = similar(B)
        for i in 1:size(B,2)
            X[:,i] = A_inv * B[:,i]
        end
        return X
    end

    function solve_factoring(A, B)
        X = similar(B)
        A = factorize(A)
        for i in 1:size(B,2)
            X[:,i] = A \ B[:,i]
        end
        return X
    end



    @btime solve_inverting($A, $B)
    @btime solve_factoring($A, $B)

    # even better, use built-in feature for multiple RHS
    @btime $A \ $B;

Triangular Matrices and Back/Forward Substitution
--------------------------------------------------

Some matrices are already in a convenient form and require no further factoring.

For example, consider solving a system with an ``UpperTriangular`` matrix,

.. code-block:: julia

    b = [1.0, 2.0, 3.0]
    U = UpperTriangular([1.0 2.0 3.0; 0.0 5.0 6.0; 0.0 0.0 9.0])

This system is especially easy to solve using `back-substitution <https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution>`_.  In particular, :math:`x_3 = b_3 / U_{33}, x_2 = (b_2 - x_3 U_{23})/U_{22}`, etc.

.. code-block:: julia

    U \ b

A ``LowerTriangular`` has similar properties and can be solved with forward-substitution.

The computational order of back-substitution and forward-substitution is :math:`O(N^2)` for dense matrices.  Those fast algorithms are a key reason that factorizaitons target triangular structures.

.. _jl_decomposition:

LU Decomposition
-------------------

The :math:`LU` decompositions finds a lower triangular :math:`L` and upper triangular :math:`U` such that :math:`L U = A`.

For a general dense matrix without any other structure (i.e. not known to be symmetric, tridiagonal, etc.) this is the standard approach to solve a system and exploit the speed of backward and forward substitution using the factorization.

The computational order of LU decomposition itself for a dense matrix is :math:`O(N^3)` - the same as Gaussian elimination, but it tends
to have a better constant term than others (e.g. half the number of operations of the QR Decomposition).  For structured
or sparse matrices, that order drops.

We can see which algorithm Julia will use for the ``\`` operator by looking at the ``factorize`` function for a given
matrix.

.. code-block:: julia

    N = 4
    A = rand(N,N)
    b = rand(N)

    Af = factorize(A)  # chooses the right factorization, LU here

In this case, it provides an :math:`L` and :math:`U` factorization (with `pivoting <https://en.wikipedia.org/wiki/LU_decomposition#LU_factorization_with_full_pivoting>`_ ).


With the factorization complete, we can solve different ``b`` right hand sides.

.. code-block:: julia

    Af \ b

.. code-block:: julia

    b2 = rand(N)
    Af \ b2

In practice, the decomposition also includes a :math:`P` is a `permutation matrix <https://en.wikipedia.org/wiki/Permutation_matrix>`_ such
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

As we saw above, this is the solution to two triangular systems, which can be efficiently done with back or forward substitution in :math:`O(N^2)` operations.

To demonstrate this, first using

.. code-block:: julia

    y = L \ b

.. code-block:: julia

    x = U \ y
    x ≈ A \ b  # Check identical

The LU decomposition also has specialized algorithms for structured matrices, such as a ``Tridiagonal``

.. code-block:: julia

    N = 1000
    b = rand(N)
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
    factorize(A) |> typeof

This factorization is the key to the performance of the ``A \ b`` in this case.  For Tridiagonal matrices, the
LU decomposition is :math:`O(N^2)`.

Finally, just as a dense matrix without any structure use an LU decomposition to solve a system,
so will the sparse solvers

.. code-block:: julia

    A_sparse = sparse(A)
    factorize(A_sparse) |> typeof  # dropping the tridiagonal structure to just become sparse

.. code-block:: julia

    benchmark_solve(A, b)
    benchmark_solve(A_sparse, b);

With sparsity, the computational order is related to the number of non-zeros rather than the size of the matrix itself.

Cholesky Decomposition
-----------------------

For real, symmetric, `positive semi-definite <https://en.wikipedia.org/wiki/Definiteness_of_a_matrix>`_ matrices, a Cholesky decomposition is a specialized example of the LU decomposition where :math:`L = U'`.


The Cholesky is directly useful on its own (e.g. :doc:`Classical Control with Linear Algebra <../time_series_models/classical_filtering>`) but it is also an efficient factorization to solve symmetric positive semi-definite systems.

As always, symmetry allows specialized algorithms.

.. code-block:: julia

    N = 500
    B = rand(N,N)
    A_dense = B' * B  # an easy way to generate a symmetric positive semi-definite matrix
    A = Symmetric(A_dense)  # flags the matrix as symmetric

    factorize(A) |> typeof

Here, the :math:`A` decomposition is `Bunch-Kaufman <https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/index.html#LinearAlgebra.bunchkaufman>`_ rather than a
Cholesky, because Julia doesn't know the matrix is positive semi-definite.  We can manually factorize with a Cholesky,

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
----------------

:ref:`Previously <qr_decomposition>`, we learned about applications of the QR decomposition to solving the linear least squares.

While in principle, the solution to the least-squares problem

.. math::

    \min_x \| Ax -b \|^2

is :math:`x = (A'A)^{-1}A'b`, in practice note that :math:`A'A` becomes dense and calculating the inverse is rarely a good idea.

The QR decomposition is a decomposition :math:`A = Q R` where :math:`Q` is an orthogonal matrix (i.e. :math:`Q'Q = Q Q' = I`) and :math:`R` is
an upper triangular matrix.

Given the  :ref:`previous derivation <qr_decomposition>` we showed that we can write the least squares problem as
the solution to

.. math::

    R x = Q' b

Where, as discussed above, the upper-triangular structure of :math:`R` can be solved easily with back substitution.

The ``\`` operator solves the linear-least squares problem whenever the given ``A`` is rectangular

.. code-block:: julia

    N = 10
    M = 3
    x_true = rand(3)

    A = rand(N,M) .+ randn(N)
    b = rand(N)
    x = A \ b

To manually use the QR decomposition in solving linear least squares:

.. code-block:: julia

    Af = qr(A)
    Q = Af.Q
    R = [Af.R; zeros(N - M, M)] # Stack with zeros
    @show Q * R ≈ A
    x = R \ Q'*b  # simplified QR solution for least squares

This stacks the ``R`` with zeros, but the more specialized algorithm would not multiply directly
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

Where the last step uses that :math:`Q^{-1} = Q'` for an orthogonal matrix.

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


Spectral Decomposition
-----------------------

A spectral decomposition, also known as an `eigendecomposition <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix>`_, finds all of the eigenvectors and eigenvalues to decompose a square matrix ``A`` such that

.. math::

    A = Q \Lambda Q^{-1}

where :math:`Q` is a matrix made of the the eigenvectors of :math:`A` as columns, and :math:`\Lambda` is a diagonal matrix of the eigenvalues.  Only square, `diagonalizable <https://en.wikipedia.org/wiki/Diagonalizable_matrix>`_ matrices have an eigendecomposition (where a matrix is not diagonalizable if it does not have a full set of linearly independent eigenvectors).

In Julia, whenever you ask for a full set of eigenvectors and eigenvalues, it  decomposes using an algorithm appropriate for the matrix type.  For example, symmetric, hermitian, or tridiagonal matrices have specialized algorithms.

To see this,

.. code-block:: julia

    A = Symmetric(rand(5, 5))  # symmetric matrices have real eigenvectors/eigenvalues
    A_eig = eigen(A)
    Λ = Diagonal(A_eig.values)
    Q = A_eig.vectors
    norm(Q * Λ * inv(Q) - A)

Keep in mind that a real matrix may have complex eigenvalues and eigenvectors, so if you attempt  to check ``Q * Λ * inv(Q) - A`` - even for a positive-definite matrix - it may not be a real number due to numerical inaccuracy.

Continuous Time Markov Chains (CTMC)
====================================

In the previous lecture on :doc:`discrete time Markov chains  <mc>`, we saw that the transition probability
between state :math:`x` and state :math:`y` was summarized by the matrix :math:`P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}`.

As a brief introduction to continuous time processes, consider the same state-space as in the discrete
case: :math:`S` a finite set with :math:`n` elements :math:`\{x_1, \ldots, x_n\}`.

A **Markov chain** :math:`\{X_t\}` on :math:`S` is a sequence of random variables on :math:`S` that have the **Markov property**.

In continuous time, the `Markov Property <https://en.wikipedia.org/wiki/Markov_property>`_ is more complicated, but intuitively is
the same as the discrete time case.

That is, knowing the current state is enough to know probabilities for future states.  Or, for realizations :math:`x(\tau)\in S, \tau \leq t`,

.. math::

    \mathbb P \{ X(t+s) = y  \,|\, X(t) = x, X(\tau) = x(\tau) \text{ for } 0 \leq \tau \leq t  \} = \mathbb P \{ X(t+s) = y  \,|\, X(t) = x\}


Heuristically, consider a time period :math:`t` and a small step forward :math:`\Delta`.  Then the probability to transition from state :math:`i` to
state :math:`j` is

.. math::

    \mathbb P \{ X(t + \Delta) = j  \,|\, X(t) \} = \begin{cases} q_{ij} \Delta + o(\Delta) & i \neq j\\
                                                                  1 + q_{ii} \Delta + o(\Delta) & i = j \end{cases}

where :math:`q_{ij}` are "intensity" parameters governing the transition rate, and :math:`o(\Delta)` is `little-o notation <https://en.wikipedia.org/wiki/Big_O_notation#Little-o_notation>`_.  That is, :math:`\lim_{\Delta\to 0} o(\Delta)/\Delta = 0`.

Just as in the discrete case, we can summarize these parameters by a :math:`N \times N` matrix, :math:`Q \in R^{N\times N}`.

Recall that in the discrete case every element is weakly positive and every row must sum to one.   Instead, with a continuous time the rows of :math:`Q` sum to zero, where the diagonal contains the negative value of jumping out of the current state.  That is

- :math:`q_{ij} \geq 0` for :math:`i \neq j`
- :math:`q_{ii} \leq 0`
- :math:`\sum_{j} q_{ij} = 0`

The :math:`Q` matrix is called the intensity matrix, or the infinitesimal generator of the Markov chain.  For example,

.. math::

    Q = \begin{bmatrix} -0.1 & 0.1  & 0 & 0 & 0 & 0\\
                        0.1  &-0.2  & 0.1 &  0 & 0 & 0\\
                        0 & 0.1 & -0.2 & 0.1 & 0 & 0\\
                        0 & 0 & 0.1 & -0.2 & 0.1 & 0\\
                        0 & 0 & 0 & 0.1 & -0.2 & 0.1\\
                        0 & 0 & 0 & 0 & 0.1 & -0.1\\
        \end{bmatrix}

In the above example, transitions only occur between adjacent states with the same intensity (except for a ``bouncing'' back of the bottom and top states).

Implementing the :math:`Q` using its tridiagonal structure

.. code-block:: julia

    using LinearAlgebra
    α = 0.1
    N = 6
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))

Here we can use the ``Tridiagonal`` to exploit the structure of the problem.

Consider a simple payoff vector :math:`r` associated with each state, and a discount rate :math:`ρ`.  Then we can solve for
the expected present discounted value in a similar way to the discrete time case.

.. math::

    \rho v = r + Q v

or rearranging slightly, solving the linear system

.. math::

    (\rho I - Q) v = r

For our example, exploiting the tridiagonal structure,

.. code-block:: julia

    r = range(0.0, 10.0, length=N)
    ρ = 0.05

    A = ρ * I - Q

Note that this :math:`A` matrix is maintaining the tridiagonal structure of the problem, which leads to an efficient solution to the
linear problem.

.. code-block:: julia

    v = A \ r

The :math:`Q` is also used to calculate the evolution of the Markov chain, in direct analogy to the :math:`ψ_{t+k} = ψ_t P^k` evolution with transition matrix :math:`P` of the discrete case.

In the continuous case, this becomes the system of linear differential equations

.. math::

    \dot{ψ}(t) = Q(t)^T ψ(t)

given the initial condition :math:`\psi(0)` and where the :math:`Q(t)` intensity matrix is allows to vary with time.  In the simplest case of a constant :math:`Q` matrix, this is a simple constant-coefficient system of linear ODEs with coefficients :math:`Q^T`.

If a stationary equilibria exists, note that :math:`\dot{ψ}(t) = 0`, and the stationary solution :math:`ψ^{*}` needs to fulfill

.. math::

    0 = Q^T ψ^{*}


Notice that this is of the form :math:`0 ψ^{*} = Q^T ψ^{*}` and hence is equivalent to finding the eigenvector associated with the :math:`\lambda = 0` eigenvalue of :math:`Q^T`.

With our example, we can calculate all of the eigenvalues and eigenvectors

.. code-block:: julia

    λ, vecs = eigen(Array(Q'))

Indeed, there is a :math:`\lambda = 0` eigenvalue, which is associated with the last column in the eigenvector.  To turn that into a probability
we need to normalize it.

.. code-block:: julia

    vecs[:,N] ./ sum(vecs[:,N])

Multiple Dimensions
--------------------

A frequent case in discretized models is dealing with Markov chains with multiple "spatial" dimensions (e.g. wealth and income).

After discretizing a process to create a Markov chain, you can always take the cartesian product of the set of states in order to
enumerate as a single state variable.

To see this, consider states :math:`i` and :math:`j` governed by infinitesimal generators :math:`Q` and :math:`A`.

.. code-block:: julia

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
    L |> Matrix  # display as a dense matrix

This provides the combined markov chain for the :math:`(i,j)` process.  To see the sparsity pattern,

.. code-block:: julia

    using Plots
    spy(L, markersize = 10)

To calculate a simple dynamic valuation, consider if the payoff of being in state :math:`(i,j)` is :math:`r_{ij} = i + 2j`

.. code-block:: julia

    r = [i + 2.0j for i in 1:N, j in 1:M]
    r = vec(r)  # vectorize it since stacked in same order

Solving the equation :math:`\rho v = r + L v`

.. code-block:: julia

    ρ = 0.05
    v = (ρ * I - L) \ r
    reshape(v, N, M)

The ``reshape`` helps to rearrange it back to being two-dimensional.


To find the stationary distribution, we calculate the eigenvalue and choose the eigenvector associated with :math:`\lambda=0` .  In this
case, we can verify it is the last one.

.. code-block:: julia

    L_eig = eigen(Matrix(L'))
    @assert norm(L_eig.values[end]) < 1E-10

    ψ = L_eig.vectors[:,end]
    ψ = ψ / sum(ψ)


Reshaping this to be two dimensional if it is helpful for visualization.

.. code-block:: julia

    reshape(ψ, N, size(A,1))

Irreducibility
--------------

As with the discrete time Markov chains, a key question is whether CTMCs are reducible, i.e. states communicate.  The problem
is isomorphic to determining if the directed graph of the Markov chain is `strongly connected <https://en.wikipedia.org/wiki/Strongly_connected_component>`_.

.. code-block:: julia

    using LightGraphs
    α = 0.1
    N = 6
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))

We can verify that it is possible to move between every state in a finite number of steps with

.. code-block:: julia

    Q_graph = DiGraph(Q)
    @show is_strongly_connected(Q_graph);  # i.e. can follow directional edges to get to every state

Alternatively, as an example of a reducible Markov chain where states :math:`1` and :math:`2` cannot jump to state :math:`3`.

.. code-block:: julia

    Q = [-0.2 0.2 0
        0.2 -0.2 0
        0.2 0.6 -0.8]
    Q_graph = DiGraph(Q)
    @show is_strongly_connected(Q_graph);

Banded Matrices
===============

A tridiagonal matrix has 3 non-zero diagonals.  The main diagonal, the first sub-diagonal (i.e. below the main diagonal) and also the first super-diagonal (i.e. above the main diagonal).

This is a special case of a more general type called a banded matrix, where the number of sub- and super-diagonals can be greater than 1.  The
total width of main-, sub-, and super-diagonals is called the bandwidth.  For example, a tridiagonal matrix has a bandwidth of 3.

A :math:`N \times N` banded matrix with bandwidth :math:`P` has about :math:`N P` nonzeros in its sparsity pattern.

These can be created directly as a dense matrix with ``diagm``.  For example, with a bandwidth of three and a zero diagonal.

.. code-block:: julia

    diagm(1 => [1,2,3], -1 => [4,5,6])

Or as a sparse matrix,

.. code-block:: julia

    spdiagm(1 => [1,2,3], -1 => [4,5,6])

Or, directly using `BandedMatrices.jl <https://github.com/JuliaMatrices/BandedMatrices.jl>`_

.. code-block:: julia

    using BandedMatrices
    BandedMatrix(1 => [1,2,3], -1 => [4,5,6])

There is also a convenience function for generating random banded matrices

.. code-block:: julia

    A = brand(7, 7, 3, 1)  # 7x7 matrix, 3 subdiagonals, 1 superdiagonal

And, of course, specialized algorithms will be used to exploit the structure when solving linear systems.  In particular, the complexity is related to the :math:`O(N P_L P_U)` for upper and lower bandwidths :math:`P`

.. code-block:: julia

    @show factorize(Symmetric(A)) |> typeof
    A \ rand(7)

The factorization algorithm uses a specialized LU decomposition for banded matrices.


.. BlockBanded and BandedBlockBanded
.. ---------------------------------

.. Taking the structured matrix concept further, we can consider examples of matrices in blocks, each of which are banded, and even
.. a matrix where each block is banded, and the blocks themselves are aligned along bands.

.. This final type is common in the discretization of multiple dimensions with continuous time processes.  For example, take the
.. example from above with 2 dimensions where the markov chain in the 2nd dimension depends on the current state in the first dimension.


.. .. code-block:: julia

..     using BandedMatrices, BlockBandedMatrices, LazyArrays
..     function markov_chain_product_banded(Q_chains, A)
..         M = size(Q_chains[1], 1)
..         N = size(A, 1)
..         Q_bands = bandwidths(Q_chains[1])

..         Qs = blockdiag(sparse.(Q_chains)...)  # create diagonal blocks of every operator
..         Qs = BandedBlockBandedMatrix(Qs, (M*ones(Int64, N), M*ones(Int64, N)), (0,0), Q_bands)

..         # construct a kronecker product of A times I_M
..         As = BandedBlockBandedMatrix(Kron(A, Eye(M)))
..         return Qs + As
..     end

..     α1 = 0.05
..     α2 = 0.15
..     α3 = 0.1
..     N = 5
..     symmetric_tridiagonal_chain(α, N) = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))
..     Q1 = symmetric_tridiagonal_chain(α1, N)
..     Q2 = symmetric_tridiagonal_chain(α2, N)
..     Q3 = symmetric_tridiagonal_chain(α3, N)
..     A = Tridiagonal([0.1, 0.1], [-0.2, -0.3, -0.2], [0.2, 0.2])
..     M = size(A,1)

..     L = markov_chain_product_banded((Q1, Q2, Q3), A);

.. The sparsity pattern shows bands of bands

.. .. code-block:: julia

..     using Plots
..     spy(sparse(L), markersize = 10)

.. As before, define a payoff function and solve the equation :math:`\rho v = r + L v`

.. .. code-block:: julia

..     r = vec([i + 2.0j for i in 1:N, j in 1:M])
..     ρ = 0.05
..     v = (ρ * I - L) \ r
..     reshape(v, N, M)

.. Or to find the stationary solution of the Markov chain, find the eigenvector associated
.. with the smallest magnitude eigenvalue (i.e. the :math:`\lambda = 0`)

.. .. code-block:: julia

..     using Arpack
..     L = sparse(L')
..     λ, ψ = eigs(L, nev=1, which=:SM)  # find smallest 1 eigenvector
..     @assert λ < 1E-8  # ensure it is the right eigenvalue/vector
..     ψ = real(ψ) ./ sum(real(ψ))
..     reshape(ψ, N, M)

.. _implementation_numerics:

Implementation Details and Performance
======================================

Recall the famous quote from Knuth: "97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%".  The most common example of premature optimization is trying to use your own mental model of a compiler while writing your code, worried about the efficiency of code and (usually incorrectly) second-guessing the compiler.

Concretely, the lessons in this section are

1. Don't worry about optimizing your code unless you need to.  Code clarity is your first-order concern.
2. If you use other people's packages, they can worry about performance and you don't need to.
3. If you absolutely need that "critical 3%" your intuition about performance is usually wrong on modern CPUs and GPUs, so let the compiler do its job.
4. Benchmarking (e.g. ``@btime``) and `profiling <https://docs.julialang.org/en/v1/manual/profile/>`_ are the tools to figure out performance bottlenecks.  If 99% of computing time is spent in 1 small function, then there is no point optimizing anything else.
5. If you benchmark to show that a particular part of the code is an issue, and you can't find another library that does a better job, then you can worry about performance.

You will rarely get to step 3, let alone step 5.

However, there is also a corollary:  "don't pessimize prematurely". That is, don't make choices that lead to poor performance without any tradeoff in improved code clarity.  For example, writing your own algorithms when a high performance algorithm exists in a package or Julia itself, or lazily making a matrix dense and carelessly dropping its structure.

Implementation Difficulty
-------------------------

Numerical analysts sometimes refer to the lowest level of code for basic operations (e.g. a dot product, matrix-matrix product, convolutions) as ``kernels``.

That sort of code is difficult to write, and performance depends on the characteristics of the underlying hardware such as the `instruction set <https://en.wikipedia.org/wiki/Instruction_set_architecture>`_ available on the particular CPU, the size of the `CPU cache <https://en.wikipedia.org/wiki/CPU_cache>`_, and the layout of arrays in memory.

Typically these operations are written in a `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ library, organized into different levels.  The levels roughly correspond to the computational order of the operations:  BLAS Level 1 are :math:`O(N)` operations such as linear products, Level 2 are :math:`O(N^2)` operations such as matrix-vector products, and Level 3 are roughly :math:`O(N^3)` such as general matrix-matrix products.

An example of a BLAS library is `OpenBLAS <https://github.com/xianyi/OpenBLAS>`_ used by default in Julia, or  the `Intel MKL <https://en.wikipedia.org/wiki/Math_Kernel_Library>`_ used in Matlab (and Julia if the ``MKL.jl`` package is installed).

On top of BLAS are `LAPACK <https://en.wikipedia.org/wiki/LAPACK>`_ operations, which are higher level kernels, such as matrix factorizations and eigenvalue algorithms, and are often in the same libraries (e.g. MKL has both BLAS and LAPACK functionality).

The details of these packages are not especially relevant, but if you are talking about performance, people will inevitably start discussing these different packages and kernels.  There are a few important things to keep in mind:

1. Leave writing kernels to the experts.  Even simple sounding algorithms can be very complicated to implement with high performance.
2. Your intuition about performance of code is probably going to be wrong.  If you use high quality libraries rather than writing your own kernels, you don't need to use your intuition.
3. Don't get distracted by the jargon or acronyms above if you are reading about performance.

Row and Column-Major Ordering
-----------------------------

There is a practical performance issue which may influence your code.  Since memory in a CPU is linear, dense matrices need to be stored by either stacking columns (called `column-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_) or rows.

The reason this matters is that compilers can generate better performance if they work in contiguous chunks of memory, and this becomes especially important with large matrices due to the interaction with the CPU cache.  Choosing the wrong order when there is no benefit in code clarity is a an example of premature pessimization.  The performance difference can be orders of magnitude in some cases, and nothing in others.

One option is to use the functions that let the compiler choose the most efficient way to traverse memory. If you need to choose the looping order yourself, then you might want to experiment with swapping whether you go through columns or rows first.  Other times, let Julia decide, i.e. ``enumerate`` and ``eachindex`` will choose the right approach.

Julia, Fortran, and Matlab all use column-major order while C/C++ and Python use row-major order.  This means that if you find an algorithm written for C/C++/Python you will sometimes need to make small changes if performance is an issue.


Digression on Allocations and Inplace Operations
-------------------------------------------------

While we have usually not considered optimizing code for performance (and focused on the choice of
algorithms instead), when matrices and vectors become large we need to be more careful.

The most important thing to avoid are excess allocations, which usually occur due to the use of
temporary vectors and matrices when they are not necessary.  Sometimes those extra temporary values
can cause enormous degredations in performance.

However, caution is suggested since
excess allocations are never relevant for scalar values, and can sometimes create faster code for
smaller matrices/vectors since it can lead to better `cache locality <https://en.wikipedia.org/wiki/Locality_of_reference>`_.

To see this, a convenient tool is the benchmarking

.. code-block:: julia

    using BenchmarkTools
    A = rand(10,10)
    B = rand(10,10)
    C = similar(A)
    function f!(C, A, B)
        D = A*B
        C .= D .+ 1
    end
    @btime f!($C, $A, $B)

The ``!`` on the ``f!`` is an informal way to say that the function is mutating, and the first arguments (``C`` here)
is by convention the modified values.

In the ``f!`` function, notice that the ``D`` is a temporary variable which is created, and then modified afterwards.  But, notice that since
``C`` is modified directly, there is no need to create the temporary ``D`` matrix.

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

Note in the output of the benchmarking, the ``f2!`` is non-allocating and is using the pre-allocated ``C`` variable directly.

Another example of this is solutions to linear equations, where for large solutions you may pre-callocate and reuse the
solution vector.

.. code-block:: julia

    A = rand(10,10)
    y = rand(10)
    z = A \ y  # creates temporary

    A = factorize(A)  # inplace requires factorization
    x = similar(y)  # pre-allocate
    ldiv!(x, A, y)  # inplace left divide, using factorization

However, if you benchmark carefully, you will see that this is sometimes slower.  Avoiding allocations is not always a good
idea - and worrying about it prior to benchmarking is premature optimization.

There are a variety of other non-allocating versions of functions.  For example,

.. code-block:: julia

    A = rand(10,10)
    B = similar(A)

    transpose!(B, A)  # non-allocating version of B = transpose(A)


Finally, a common source of unnecessary allocations is when taking slices or portions of
matrices.  For example, the following allocates a new matrix ``B`` and copies the values.

.. code-block:: julia

    A = rand(5,5)
    B = A[2,:]  # extract a vector

To see these are different matrices, note that

.. code-block:: julia

    A[2,1] = 100.0
    @show A[2,1]
    @show B[1];

Instead of allocating a new matrix, you can take a ``view`` of a matrix, which provides an
appropriate ``AbstractArray`` type that doesn't allocate new memory with the ``@view`` matrix.

.. code-block:: julia

    A = rand(5,5)
    B = @view A[2,:]  #  does not copy the data

    A[2,1] = 100.0
    @show A[2,1]
    @show B[1];

But, again, you will often find that doing ``@view`` leads to slower code.  Benchmark
instead, and generally rely on it for large matrices and for contiguous chunks of memory (e.g. a column rather than a row).

Exercises
==============

Exercise 1
------------

This exercise is for a practice on writing low-level routines (i.e. "kernels"), and to hopefully convince you to leave low-level code to the experts.

The formula for matrix multiplication is deceptively simple.  For example, with the product of square matrices :math:`C = A B` of size :math:`N \times N`, the :math:`i,j` element of :math:`C` is

.. math::

    C_{ij} = \sum_{k=1}^N A_{ik} B_{kj}

Alternatively, you can take a row :math:`A_{i,:}` and column :math:`B_{:, j}` and use an inner product

.. math::

    C_{ij} = A_{i,:} \cdot B_{:,j}

Note that the inner product in a discrete space is simply a sum, and has the same complexity as the sum (i.e. :math:`O(N)` operations).

For a dense matrix without any structure using a naive multiplication algorithm, this also makes it clear why the complexity is :math:`O(N^3)`: you need to evaluate it for :math:`N^2` elements in the matrix and do an :math:`O(N)` operation each time.

For this exercise, implement matrix multiplication yourself and compare performance in a few permutations.

#. Use the built-in function in Julia (i.e.``C = A * B`` or, for a better comparison, the inplace version ``mul!(C, A, B)`` which works with preallocated data)
#. Loop over each :math:`C_{ij}` by the row first (i.e. the ``i`` index) and use a ``for`` loop for the inner product
#. Loop over each :math:`C_{ij}` by the column first (i.e. the ``j`` index) and use a ``for`` loop for the inner product
#. Do the same but use the ``dot`` product instead of the sum.
#. Choose your best implementation of these, and then for matrices of a few different sizes ``N=10``, ``N=1000``, etc. and compare the ratio of performance of your best implementation to the built in BLAS library.

A few more hints:

- You can just use random matrices, e.g. ``A = rand(N, N)``, etc.
- For all of them, preallocate the :math:`C` matrix beforehand with ``C = similar(A)`` or something equivalent.
- To compare performance, put your code in a function and use ``@btime`` macro to time it.

Exercise 2a
--------------

Here we will calculate the evolution of the pdf of a discrete time Markov Chain, :math:`\psi_t` given the initial condition :math:`\psi_0`.

Start with a simple symmetric tridiagonal matrix

.. code:: julia

    N = 100
    A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2)])
    A_adjoint = A';

1. Pick some large ``T`` and use the initial condition :math:`\psi_0 = \begin{bmatrix} 1 & 0 & \ldots & 0\end{bmatrix}`
2. Write code to calculate :math:`\psi_t` to some :math:`T` by iterating the map for each :math:`t`, i.e.

.. math::

    \psi_{t+1} = A' \psi_t

3. What is the computational order of calculating  :math:`\psi_T` using this iteration approach :math:`T < N`?
4. What is the computational order of :math:`(A')^T = (A' \ldots A')` and then :math:`\psi_T = (A')^T \psi_0` for :math:`T < N`?
5. Benchmark calculating :math:`\psi_T` with the iterative calculation above as well as the direct :math:`\psi_T = (A')^T \psi_0` to see which is faster.  You can take the matrix power with just ``A_adjoint^T``, which uses specialized algorithms faster and more accurate than repeated matrix multiplication (but with the same computational order).
6. Check the same if :math:`T = 2 N`

*Note:* The algorithm used in Julia to take matrix powers  depends on the matrix structure, as always.  In the symmetric case, it can use an eigendecomposition, whereas with a general dense matrix it uses `squaring and scaling <https://doi.org/10.1137/090768539>`_.

Exercise 2b
--------------

With the same setup as Exercise 2a, do an `eigendecomposition <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix>`_ of ``A_transpose``.  That is, use ``eigen`` to factorize the adjoint :math:`A' = Q \Lambda Q^{-1}` where :math:`Q` the matrix of eigenvectors and :math:`\Lambda` the diagonal matrix of eigenvalues.  Calculate :math:`Q^{-1}` from the results.

Use the factored matrix to calculate the sequence of :math:`\psi_t = (A')^t \psi_0` using the relationship

.. math::

    \psi_t = Q \Lambda^t Q^{-1} \psi_0

Where matrix powers of diagonal matrices are simply the element-wise power of each element.

Benchmark the speed of calculating the sequence of :math:`\psi_t` up to ``T = 2N`` using this method.  In principle, the factorization and easy calculation of the power should give you benefits compared to simply iterating the map as we did in Exercise 2a.  Explain why it does or does not using computational order of each approach.
