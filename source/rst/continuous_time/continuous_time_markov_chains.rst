.. _seir_sciml:

.. include:: /_static/includes/header.raw

.. highlight:: julia

*******************************************************************
:index:`Continuous Time Markov Chains with COVID 19 Applications`
*******************************************************************

.. contents:: :depth: 2

Overview
=============

Coauthored with Chris Rackauckas

This lecture develops the theory of using continuous time markov chains, in the same spirit as :doc:`Discrete Time Markov Chains <../tools_and_techniques/finite_markov>` and with some of the same applications as that of  :doc:`Discrete State Dynamic Programming <../dynamic_programming/discrete_dp>`

Here, we will also explore where the ODE approximation of :doc:`Modeling COVID 19 with (Stochastic) Differential Equations <../continuous_time/seir_model_sde>`, comes from


Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics, Random, SparseArrays

.. code-block:: julia
    :class: Test

    using Test # Put this before any code in the lecture.

In addition, we will be exploring packages within the `SciML ecosystem <https://github.com/SciML/>`__ and
others covered in previous lectures 

.. code-block:: julia

    using OrdinaryDiffEq, StochasticDiffEq, DiffEqJump, DiffEqFlux, DiffEqBayes
    using Parameters, StaticArrays, Plots


ROUGH NOTES
==============

Notes from John:-
- matrix exponentials
- Q matrices and their relationship to Markov matrices (via matrix exponential, Kolmogorov fwd and backward equations)
- exponential clock interpretation of Q matrices
- ergodicity
- applications (inventory dynamics?)

Lets of ideas from Chris in https://github.com/QuantEcon/lecture-source-jl/issues/859#issuecomment-632998856

The goal for that might me to building up from a Poisson process to show how the SDE approximation works Gillespie's SSA, and then show how in the limit of a
large number of particles we get the ODE approximation that economists are familiar with.

I also think that getting more familiar with the SDE approximation of a discrete number of agents could have other applciations in economics (e.g. search)


Continuous-Time Markov Chains (CTMCs)
=====================================

In the lecture on :doc:`discrete-time Markov chains  <mc>`, we saw that the transition probability
between state :math:`x` and state :math:`y` was summarized by the matrix :math:`P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}`.

As a brief introduction to continuous time processes, consider the same state space as in the discrete
case: :math:`S` is a finite set with :math:`n` elements :math:`\{x_1, \ldots, x_n\}`.

A **Markov chain** :math:`\{X_t\}` on :math:`S` is a sequence of random variables on :math:`S` that have the **Markov property**.

In continuous time, the `Markov Property <https://en.wikipedia.org/wiki/Markov_property>`_ is more complicated, but intuitively is
the same as the discrete-time case.

That is, knowing the current state is enough to know probabilities for future states.  Or, for realizations :math:`x(\tau)\in S, \tau \leq t`,

.. math::

    \mathbb P \{ X(t+s) = y  \,|\, X(t) = x, X(\tau) = x(\tau) \text{ for } 0 \leq \tau \leq t  \} = \mathbb P \{ X(t+s) = y  \,|\, X(t) = x\}


Heuristically, consider a time period :math:`t` and a small step forward, :math:`\Delta`.  Then the probability to transition from state :math:`i` to
state :math:`j` is

.. math::

    \mathbb P \{ X(t + \Delta) = j  \,|\, X(t) \} = \begin{cases} q_{ij} \Delta + o(\Delta) & i \neq j\\
                                                                  1 + q_{ii} \Delta + o(\Delta) & i = j \end{cases}

where the :math:`q_{ij}` are "intensity" parameters governing the transition rate, and :math:`o(\Delta)` is `little-o notation <https://en.wikipedia.org/wiki/Big_O_notation#Little-o_notation>`_.  That is, :math:`\lim_{\Delta\to 0} o(\Delta)/\Delta = 0`.

Just as in the discrete case, we can summarize these parameters by an :math:`N \times N` matrix, :math:`Q \in R^{N\times N}`.

Recall that in the discrete case every element is weakly positive and every row must sum to one.   With continuous time, however, the rows of :math:`Q` sum to zero, where the diagonal contains the negative value of jumping out of the current state.  That is,

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

In the above example, transitions occur only between adjacent states with the same intensity (except for a ``bouncing back'' of the bottom and top states).

Implementing the :math:`Q` using its tridiagonal structure

.. code-block:: julia

    using LinearAlgebra
    α = 0.1
    N = 6
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))

Here we can use ``Tridiagonal`` to exploit the structure of the problem.

Consider a simple payoff vector :math:`r` associated with each state, and a discount rate :math:`ρ`.  Then we can solve for
the expected present discounted value in a way similar to the discrete-time case.

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

The :math:`Q` is also used to calculate the evolution of the Markov chain, in direct analogy to the :math:`ψ_{t+k} = ψ_t P^k` evolution with the transition matrix :math:`P` of the discrete case.

In the continuous case, this becomes the system of linear differential equations

.. math::

    \dot{ψ}(t) = Q(t)^T ψ(t)

given the initial condition :math:`\psi(0)` and where the :math:`Q(t)` intensity matrix is allowed to vary with time.  In the simplest case of a constant :math:`Q` matrix, this is a simple constant-coefficient system of linear ODEs with coefficients :math:`Q^T`.

If a stationary equilibrium exists, note that :math:`\dot{ψ}(t) = 0`, and the stationary solution :math:`ψ^{*}` needs to satisfy

.. math::

    0 = Q^T ψ^{*}


Notice that this is of the form :math:`0 ψ^{*} = Q^T ψ^{*}` and hence is equivalent to finding the eigenvector associated with the :math:`\lambda = 0` eigenvalue of :math:`Q^T`.

With our example, we can calculate all of the eigenvalues and eigenvectors

.. code-block:: julia

    λ, vecs = eigen(Array(Q'))

Indeed, there is a :math:`\lambda = 0` eigenvalue, which is associated with the last column in the eigenvector.  To turn that into a probability,
we need to normalize it.

.. code-block:: julia

    vecs[:,N] ./ sum(vecs[:,N])

Multiple Dimensions
--------------------

A frequent case in discretized models is dealing with Markov chains with multiple "spatial" dimensions (e.g., wealth and income).

After discretizing a process to create a Markov chain, you can always take the Cartesian product of the set of states in order to
enumerate it as a single state variable.

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

This provides the combined Markov chain for the :math:`(i,j)` process.  To see the sparsity pattern,

.. code-block:: julia

    using Plots
    gr(fmt = :png);
    spy(L, markersize = 10)

To calculate a simple dynamic valuation, consider whether the payoff of being in state :math:`(i,j)` is :math:`r_{ij} = i + 2j`

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
case, we can verify that it is the last one.

.. code-block:: julia

    L_eig = eigen(Matrix(L'))
    @assert norm(L_eig.values[end]) < 1E-10

    ψ = L_eig.vectors[:,end]
    ψ = ψ / sum(ψ)


Reshape this to be two-dimensional if it is helpful for visualization.

.. code-block:: julia

    reshape(ψ, N, size(A,1))

Irreducibility
--------------

As with the discrete-time Markov chains, a key question is whether CTMCs are reducible, i.e., whether states communicate.  The problem
is isomorphic to determining whether the directed graph of the Markov chain is `strongly connected <https://en.wikipedia.org/wiki/Strongly_connected_component>`_.

.. code-block:: julia

    using LightGraphs
    α = 0.1
    N = 6
    Q = Tridiagonal(fill(α, N-1), [-α; fill(-2α, N-2); -α], fill(α, N-1))

We can verify that it is possible to move between every pair of states in a finite number of steps with

.. code-block:: julia

    Q_graph = DiGraph(Q)
    @show is_strongly_connected(Q_graph);  # i.e., can follow directional edges to get to every state

Alternatively, as an example of a reducible Markov chain where states :math:`1` and :math:`2` cannot jump to state :math:`3`.

.. code-block:: julia

    Q = [-0.2 0.2 0
        0.2 -0.2 0
        0.2 0.6 -0.8]
    Q_graph = DiGraph(Q)
    @show is_strongly_connected(Q_graph);
