.. _iterative_methods_sparsity:

.. include:: /_static/includes/header.raw


*****************************************
Conditioning and Iterative Methods
*****************************************

.. contents:: :depth: 2


Overview
============

This lecture takes the structure of :doc:` numerical methods for linear algebra <numerical_linear_algebra>` and builds further
towards working with large, sparse matrices.  In the process, we will examine foundational numerical analysis such as
ill-conditioned matrices.

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics, BenchmarkTools, Random
    Random.seed!(42);  # seed random numbers for reproducibility

Ill-conditioned Matrices
========================

An important consideration in numerical linear algebra, and iterative methods in general is the `condition number <https://en.wikipedia.org/wiki/Condition_number#Matrices>`_.

An ill-conditioned matrix is one where the basis of eigenvectors are close to, but not exactly, collinear.  While this poses no problem on pen and paper,
or with infinite precision numerical methods, it is an important issue in practice for two reasons

1. Ill-conditioned matrices introduce numerical errors roughly in proportion to the base-10 log of the condition number.
2. The convergence speed of many iterative methods is based on the spectral properties (e.g. the basis formed by the eigenvectors), and hence ill-conditioned systems can converge slowly.

The solutions to these problems are to

- be careful with operations which introduce error based on the condition number (e.g. matrix inversions when the condition number is high)
- where possible, choose alternative representations which have less collinearity (e.g. an orthogonal polynomial basis rather than a monomial one)
- for iterative methods, use a preconditioner, which changes the spectral properties to increase acceleration speed

Condition Number
----------------

First, lets define the condition number and example it

.. math::

    \kappa(A) = \|A\| \|A^{-1}\|

where you can use the Cauchy–Schwarz inequality to show that :math:`\kappa(A) \geq 1`.  You can choose any norm, but the 2-norm is a good default.

First, a warning on calculations: calculating the condition number for a matrix can be an expensive operation (as would calculating a determinant)
and should be thought of as roughly equivalent to doing an eigendecomposition.  So use it for detective work judiciously.  

Lets look at the condition number of a few matrices using the ``cond`` function (which allows a choice of the norm, but we stick with the 2-norm).

.. code-block:: julia

    A = I(2)
    cond(A)

Here we see an example of the best-conditioned matrix, the identity matrix with its completely orthonormal basis, is 1.

On the other hand, notice that

.. code-block:: julia

    ϵ = 1E-6
    A = [1.0 0.0
         1.0 ϵ]
    cond(A)

Has a condition number of close to 100,000 - and hence (taking the base 10 log) you would expect to be introducing numerical errors of around 6 digits if you
are not careful.  For example, note that the inverse has both extremely large and extremely small numbers

.. code-block:: julia

    inv(A)

Since we know that the determinant of nearly collinear matrices is close to zero, this shows another symptom of poor conditioning

.. code-block:: julia

    det(A)

However, be careful since the determinant has a scale, while the condition number is dimensionless.  That is

.. code-block:: julia

    @show det(1000 * A)
    @show cond(1000 * A);

In that case, the determinant of ``A`` is 1, while the condition number is unchanged.  This example also provides some
intuition that ill-conditioned matrices typically occur when a matrix has radically different scales (e.g. contains both ``1`` and ``1E-6``, or ``1000`` and ``1E-3``).  This can occur frequently with both function approximation and linear-least squares. 

Why Monomial Basis are a Bad Idea
---------------------------------

A classic example of poorly conditioned matrices is using a monomial basis of a polynomial with interpolation.

Take a grid of points, :math:`x_0, \ldots x_N` and values :math:`y_0, \ldots y_N` where we want to calculate the
interpolating polynomial.

If we were to use the "obvious" polynomial basis, then the calculation is to calculate the coefficients :math:`c_1, \ldots c_n` where

.. math::

    P(x) = \sum_{i=0}^N c_i x^i 

To solve for the coefficients, we notice that this is a simple system of equations

.. math::

    \begin{array}
        y_0 = c_0 + c_1 x_0 + \ldots c_N x_0^N\\
        \ldots\\
        y_N = c_0 + c_1 x_N + \ldots c_N x_N^N
    \end{array}

Or, stacking as matrices and vectors :math:`c = \begin{bmatrix} c_0 & \ldots & c_N\end{bmatrix}, y = \begin{bmatrix} y_0 & \ldots & y_N\end{bmatrix}` and 

.. math::

    A = \begin{bmatrix} 1 & x_0 & x_0^2 & \ldots &x_0^N\\
                        \vdots & \vdots & \vdots & \vdots & \vdots \\
                        1 & x_N & x_N^2 & \ldots & x_N^N 
        \end{bmatrix}

We can then calculate the interpolating coefficients as the solution to

.. math::

    A c = y

Lets see this in operation

.. code-block:: julia

    N = 5
    f(x) = exp(x)
    x = range(0.0, 10.0, length = N+1)
    y = f.(x)  # generate some data to interpolate

    A = [x_i^n for x_i in x, n in 0:N]
    A_inv = inv(A)
    c = A_inv * y
    norm(A * c - f.(x), Inf)

The final step is to loop.  The Inf-norm (i.e. maximum difference) of the interpolation errors is around ``1E-9`` which
is reasonable for many problems.

But note that with :math:`N=5` the condition number is already into the tens of thousands.

.. code-block:: julia

    cond(A)

What if we increase the degree of the polynomial with the hope of increasing the precision of the
interpolation?

.. code-block:: julia

    N = 10
    f(x) = exp(x)
    x = range(0.0, 10.0, length = N+1)
    y = f.(x)  # generate some data to interpolate

    A = [x_i^n for x_i in x, n in 0:N]
    A_inv = inv(A)
    c = A_inv * y
    norm(A * c - f.(x), Inf)

Here, we see that the the increasing precision is backfiring and by going to the modest basis of 10 we have
introduced an error of about ``1E-6``, even at the interpolation points.

This blows up quickly

.. code-block:: julia

    N = 20
    f(x) = exp(x)
    x = range(0.0, 10.0, length = N+1)
    y = f.(x)  # generate some data to interpolate

    A = [x_i^n for x_i in x, n in 0:N]
    A_inv = inv(A)
    c = A_inv * y
    norm(A * c - f.(x), Inf)

To see the source of the issue, we can check to see condition number is astronomical.

.. code-block:: julia

    cond(A)

At this point, you should be suspicious of the use of ``inv(A)`` since we have considered solving
linear systems by taking the inverse as verboten.  Indeed, this didn't help and we see the
error drop dramatically.

.. code-block:: julia

    c = A \ y
    norm(A * c - f.(x), Inf)

But an error of ``1E-10`` at the interpolating nodes themselves can be an issue in many applications, and if you increase ``N``
then the error will become non-trivial quickly - even without taking the inverse.

At the heart of the issue is that the monomial basis leads to a `Vandermonde_matrix <https://en.wikipedia.org/wiki/Vandermonde_matrix>`_ which
is especially ill-conditioned.  

We can also understand a separate type of errors called `Runge's Phenomena <https://en.wikipedia.org/wiki/Runge%27s_phenomenon>`_.    It is an important
issue in approximation theory, albeit not one driven by numerical error themselves.  

It turns out that using a uniform grid of points is close to the worst possible choice of interpolation nodes for a polynomial approximation.  This phenomena is can be seen with the interpolation of the seemingly innocuous Runge's function, :math:`g(x) = \frac{1}{1 + 25 x^2}`.


Let us interpolate this function using the monomial basis above to find the :math:`c_i` such that

.. math::

    \frac{1}{(1 + 25 x^2} \approx \sum_{i=0}^N c_i x^i 

Implementing, where we know that for ``N=5`` the numerical error from being ill-conditioning is manageable, we see the
approximation has large errors at the corners. 

.. code-block:: julia

    using Plots
    N_display = 100
    g(x) = 1/(1 + 25x^2)
    x_display = range(-1, 1, length = N_display)
    y_display = g.(x_display)

    # interpolation 
    N = 5
    x = range(-1.0, 1.0, length = N+1)
    y = g.(x)
    A_5 = [x_i^n for x_i in x, n in 0:N] 
    c_5 = A_5 \ y

    # use the coefficients to evaluate on x_display grid
    B_5 = [x_i^n for x_i in x_display, n in 0:N]   # calculate monomials for display grid
    y_5 = B_5 * c_5  # calculates for each in x_display_grid
    plot(x_display, y_5, label = "N=5")
    plot!(x_display, y_display, w = 3, label = "Runge's")

This has the hallmark oscillations near the boundaries of Runge's Phenomena.  You might guess that increasing the number
of grid points and order of the polynomial will lead to better approximations

.. code-block:: julia

    N = 9
    x = range(-1.0, 1.0, length = N+1)
    y = g.(x)
    A_9 = [x_i^n for x_i in x, n in 0:N]
    c_9 = A_9 \ y

    # use the coefficients to evaluate on x_display grid
    B_9 = [x_i^n for x_i in x_display, n in 0:N]   # calculate monomials for display grid
    y_9 = B_9 * c_9  # calculates for each in x_display_grid
    plot(x_display, y_9, label = "N=9")
    plot!(x_display, y_display, w = 3, label = "Runge's")

Instead, we see that while the approximation is better near ``x=0``, the oscillations near the boundaries have become worse.

Using an Orthogonal Polynomial Basis
------------------------------------ 

We can minimize the numerical issues of an ill-conditioned matrix by choosing a different basis for the polynomials.

For example, with `Chebyshev polymomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_, which form an orthonormal basis, we can form precise high-order approximations, with very little numerical error 

.. code-block:: julia

    using ApproxFun
    N = 10000
    S = Chebyshev(0.0..10.0)  # form chebyshev basis
    x = points(S, N)  # chooses better grid points, but that could be modified
    y = f.(x)
    f_approx = Fun(S,ApproxFun.transform(S,y))  # transform fits the polynomial
    norm(f_approx.(x) - exp.(x), Inf)

Besides the use of a different polynomial basis, we are approximating at different nodes (i.e. `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_).  Interpolation with Chebyshev polynomials at the Chebyshev nodes ends up minimizing (but not eliminating) Runge's Phenomena.

To summarize the analysis,

1. Check the condition number on systems you suspect might be ill-conditioned (based on intuition of collinearity).
2. If you are working with ill-conditioned matrices, be especially careful not to take inverse.
3. Avoid a monomial polynomial basis.  Instead, orthogonal polynomials (e.g. Chebyshev or Lagrange) which are orthogonal under the inner product, or non-global basis such as cubic-splines.
4. If possible, avoid using a uniform grid for interpolation and approximation and choose nodes appropriate for the basis.

However, sometimes you can't avoid ill-conditioned matrices. This is especially common with discretization of PDEs and with linear-least squares.

Iterative Algorithms for Linear Systems
=======================================

As before, consider solving the equation 

.. math::

    A x = b

where we will maintain a solution that, if :math:`A` is square, there is a unique solution.  However, we will now
focus on cases where :math:`A` is both massive, sparse (e.g. potentially billions of equations), and sometimes ill-conditioned.  

While this may seem excessive, it occurs in practice due to the curse of dimensionality, discretizations
of PDEs, and when working with big or network data.

The methods in the previous lectures (e.g. factorization and the related Gaussian elimination) are called direct methods, and able 
- in theory - to converge to the exact solution in a finite number of steps while working with the matrix.  As we saw before, solving a dense linear
system without any structure takes :math:`O(N^3)` operations, while a sparse system depends on the number of non-zeros.

Instead, iterative solutions start with a guess on a solution and iterate until until asymptoptic convergence.  The benefit will be that
each iteration uses a much lower order operation (e.g. an :math:`O(N^2)` matrix-vector product) which will make it possible to both: (1)
solve much larger systems, even if done less precisely and (2) define linear operators in terms of the matrix-vector products directly; and (3) find solutions
in progress prior to the completion of all algorithm steps.

So, rather than always thinking of linear operators as being matrices, we will consider linear operators that may or may not fit in memory (leading to "matrix-free methods"), but implement a left-multiply ``*`` operator for vectors.

There are two types of iterative methods we will consider:  first are stationary methods which iterate on a map, in a similar way to fixed point problems (and which sometimes have similar contraction mapping requirements) and the second are krylov methods which iteratively solve using a basis of the matrices.

For our main examples, lets solve the valuation of the continuous time markov chain from the previous section.  That is, given a payoff vector :math:`r`, a
discount rate :math:`\rho`, and the infinitesimal generator of the markov chain :math:`Q`, solve the equation


.. math::

    \rho v = r + Q v 

With the sizes and types of matrices here, iterative methods are inappropriate in practice, but it will help us understand
the characteristics of convergence, and how they relate to matrix conditioning.

Stationary Methods
--------------------

First, we will solve with a direct methods, which will give the solution to machine precision.  

.. code:: julia

    using LinearAlgebra, IterativeSolvers, Statistics
    α = 0.1
    N = 100
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))

    r = range(0.0, 10.0, length=N)
    ρ = 0.05

    A = ρ * I - Q
    v_direct = A \ r
    mean(v_direct)

Without proof, consider given the discount rate of :math:`\rho > 0` this problem could be setup as a contraction for solving the Bellman
equation through methods like value function iteration.

The condition we will examine here is called `**diagonal dominance** <https://en.wikipedia.org/wiki/Diagonally_dominant_matrix>`_.

.. math::

    |A_{ii}| \geq \sum_{j\neq i} |A_{ij}| \quad\text{for all } i = 1\ldots N

That is, for every row, the diagonal is weakly greater in absolute value than the sum of all of the other elements in the row.  In cases
where it is strictly greater, we say that the matrix is strictly diagonally dominant.

With our example, given that :math:`Q` is the infinitesimal generator of a markov chain, we know that each row sums to 0, and hence
it is weakly diagonally dominant.

However, notice that when :math:`\rho > 0`,  :math:`A = ρ * I - Q` makes the matrix strictly diagonally dominant.

Jacobi Iteration
----------------

For matrices that are **strictly diagonally dominant**, you can prove that a simple decomposition and iteration procedure
will converge. 

To solve a system :math:`A x = b`, split the matrix :math:`A` into its diagonal and off-diagonals.  That is,

.. math:: 

    A = D + R 

where

.. math::

    D = \begin{bmatrix} A_{11} & 0 & \ldots & 0\\
                        0    & A_{22} & \ldots & 0\\
                        \vdots & \vdots & \vdots & \vdots\\
                        0 & 0 &  \ldots & 0 A_{NN}

and

.. math::

    D = \begin{bmatrix} 0 & A_{12}  & \ldots & A_{1N} \\
                        A_{21}    & 0 & \ldots & A_{2N} \\
                        \vdots & \vdots & \vdots & \vdots\\
                        A_{N1}  & A_{N2}  &  \ldots & 0
        \end{bmatrix}

TEXTTBF:::: TODO!!!s

Start with a :math:`v` guess, 

.. code-block:: julia

    using IterativeSolvers
    v = zeros(N)
    jacobi!(v, A, r, maxiter = 1000, log=true)
    @show norm(v - v_direct, Inf)

    using IterativeSolvers
    v = zeros(N)
    jacobi!(v, A, r, maxiter = 40)
    @show norm(v - v_direct, Inf)

    v = zeros(N)
    gauss_seidel!(v, A, r, maxiter = 40)
    @show norm(v - v_direct, Inf);

    v = zeros(N)
    sor!(v, A, r, 1.1, maxiter = 40)
    @show norm(v - v_direct, Inf);

    using LinearAlgebra, SparseArrays, IterativeSolvers
    A = I + rand(100,100)  # the real example is assymetric...
    b = rand(100)
    x = similar(b)
    #gmres!(x, A, b, Pl = Identity(), log=true, maxiter = 1000)
    A_lu = lu(A)
    gmres!(x, A, b, Pl = A_lu, log=true)  # do my own preconditioning

    using IncompleteLU
    P = ilu(sparse(A), τ = 0.1)
    gmres!(x, A, b, Pl = P, log=true)

..  
.. ------

.. There are many algorithms which exploit matrix symmetry and positive-definitness (e.g. the conjugate gradient method) or simply symmetric/hermitian (e.g. MINRES).

.. On the other hand, if you do not have any structure to your sparse matrix, then GMRES is a good approach.

.. To experiment with these methods, we will use our ill-conditioned interpolation problem with a monomial basis

.. .. code-block:: julia

..     using IterativeSolvers
    
..     N = 10
..     f(x) = exp(x)
..     x = range(0.0, 10.0, length = N+1)
..     y = f.(x)  # generate some data to interpolate
..     A = [x_i^n for x_i in x, n in 0:N]
 
..     using IterativeSolvers
..     c = zeros(N+1)  # initial guess required for iterative solutions
..     results = gmres!(c, A, y)
..     println("cond(A) = $(cond(A)), converged in $(length(results)) iteration with norm error $(norm(A*c - y, Inf))")

.. In this case, both the error and the number of iterations are reasonable.  However, as ``N`` increases
.. the method fails.

.. .. code-block:: julia

..     N = 30
..     f(x) = exp(x)
..     x = range(0.0, 10.0, length = N+1)
..     y = f.(x)  # generate some data to interpolate
..     A = [x_i^n for x_i in x, n in 0:N]
 
..     c = zeros(N+1)  # initial guess required for iterative solutions
..     results = gmres!(c, A, y)
..     println("cond(A) = $(cond(A)), converged in $(length(results)) iterations with norm error $(norm(A*c - y, Inf))")

.. Iterative Methods for Eigensystems
.. ====================================

.. When you use ``eigen`` on a matrix, it calculates the full spectral decomposition, providing all of the eigenvalues and eigenvectors.

.. While sometimes this is necessary, a spectral decomposition of a dense, unstructured matrix is one of the costliest :math:`O(N^3)` operations (i.e., it has
.. one of the largest constants).  For large matrices it is often infeasible. 

.. Luckily, you frequently only need a few or even a single eigenvector/eigenvalue, which enables a different set of algorithms.

.. For example, in the case of a discrete time markov chain, to find the stationary distribution we are looking for the
.. eigenvector associated with the eigenvalue of 1.  As usual, a little linear algebra goes a long way.

.. From the `Perron-Frobenius theorem <https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem#Stochastic_matrices>`_, the largest eigenvalue of an irreducible stochastic matrix is 1 - the same eigenvalue we are looking for.

.. Iterative methods for solving eigensystems allow targeting the smallest magnitude, largest magnitude, and many others.  The easiest library
.. to use is `Arpack.jl <https://julialinearalgebra.github.io/Arpack.jl/latest/>`_.  

.. .. code-block:: julia

..     using Arpack, LinearAlgebra
..     N = 1000
..     A = Tridiagonal([fill(0.1, N-2); 0.2], fill(0.8, N), [0.2; fill(0.1, N-2);])
..     A_adjoint = A'

..     λ, ϕ = eigs(A_adjoint, nev=1, which=:LM, maxiter=1000)  # Find 1 of the largest magnitude eigenvalue
..     ϕ = real(ϕ) ./ sum(real(ϕ))
..     λ

.. Indeed, the ``λ`` is equal to ``1``.   Hint: if you get errors, increase ``maxiter``.

.. Similarly, for a continuous time Markov Chain, to find the stationary distribution we are looking for the eigenvector associated with ``λ = 0`, which
.. must be the smallest absolute magnitude.

.. With our multi-dimensional CTMC in the previous section

.. .. code-block:: julia

..    using SparseArrays
..     function markov_chain_product(Q, A)
..         M = size(Q, 1)
..         N = size(A, 1)
..         Q = sparse(Q)
..         Qs = blockdiag(fill(Q, N)...)  # create diagonal blocks of every operator
..         As = kron(A, sparse(I(M)))
..         return As + Qs
..     end

..     α = 0.1
..     N = 4
..     Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))
..     A = sparse([-0.1 0.1
..         0.2 -0.2])
..     M = size(A,1)
..     L = markov_chain_product(Q, A)
..     L_adjoint = L';

.. In this case, Arpack.jl does not do well with the singular matrix of a CTMC.  We can use another package
.. for similar methods called `KrylovKit.jl <https://jutho.github.io/KrylovKit.jl/latest/man/eig/>`_

.. .. code-block:: julia

..     using KrylovKit
..     λ, ϕ = KrylovKit.eigsolve(L_adjoint, 1, :SR)  # smallest absolute value
..     reshape(real(ϕ[end]), N, size(A,1))
..     ϕ = real(ϕ) ./ sum(real(ϕ))
..     ϕ = reshape(ϕ, N, size(A,1))
..     λ

