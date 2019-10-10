.. _mc:

.. include:: /_static/includes/header.raw

.. highlight:: julia

************************************
:index:`Finite Markov Chains`
************************************

.. contents:: :depth: 2

Overview
=============

Markov chains are one of the most useful classes of stochastic processes, being

* simple, flexible and supported by many elegant theoretical results

* valuable for building intuition about random dynamic models

* central to quantitative modeling in their own right

You will find them in many of the workhorse models of economics and finance

In this lecture we review some of the theory of Markov chains

We will also introduce some of the high quality routines for working with Markov chains available in `QuantEcon.jl <http://quantecon.org/julia_index.html>`__

Prerequisite knowledge is basic probability and linear algebra

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics
    using Distributions, Plots, Printf, QuantEcon, Random
    gr(fmt = :png);

Definitions
==============

The following concepts are fundamental

.. _finite_dp_stoch_mat:

:index:`Stochastic Matrices`
-----------------------------

.. index::
    single: Finite Markov Chains; Stochastic Matrices

A **stochastic matrix** (or **Markov matrix**)  is an :math:`n \times n` square matrix :math:`P`
such that

#. each element of :math:`P` is nonnegative, and

#. each row of :math:`P` sums to one

Each row of :math:`P` can be regarded as a probability mass function over :math:`n` possible outcomes

It is too not difficult to check [#pm]_ that if :math:`P` is a stochastic matrix, then so is the :math:`k`-th power :math:`P^k` for all :math:`k \in \mathbb N`

:index:`Markov Chains`
-----------------------------

.. index::
    single: Finite Markov Chains

There is a close connection between stochastic matrices and Markov chains

To begin, let :math:`S` be a finite set with :math:`n` elements :math:`\{x_1, \ldots, x_n\}`

The set :math:`S` is called the **state space** and :math:`x_1, \ldots, x_n` are the **state values**

A **Markov chain** :math:`\{X_t\}` on :math:`S` is a sequence of random variables on :math:`S` that have the **Markov property**

This means that, for any date :math:`t` and any state :math:`y \in S`,

.. math::
    :label: fin_markov_mp

    \mathbb P \{ X_{t+1} = y  \,|\, X_t \}
    = \mathbb P \{ X_{t+1}  = y \,|\, X_t, X_{t-1}, \ldots \}

In other words, knowing the current state is enough to know probabilities for future states

In particular, the dynamics of a Markov chain are fully determined by the set of values

.. math::
    :label: mpp

    P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}
    \qquad (x, y \in S)

By construction,

* :math:`P(x, y)` is the probability of going from :math:`x` to :math:`y` in one unit of time (one step)

* :math:`P(x, \cdot)` is the conditional distribution of :math:`X_{t+1}` given :math:`X_t = x`

We can view :math:`P` as a stochastic matrix where

.. math::

    P_{ij} = P(x_i, x_j)
    \qquad 1 \leq i, j \leq n

Going the other way, if we take a stochastic matrix :math:`P`, we can generate a Markov
chain :math:`\{X_t\}` as follows:

* draw :math:`X_0` from some specified distribution

* for each :math:`t = 0, 1, \ldots`, draw :math:`X_{t+1}` from :math:`P(X_t,\cdot)`

By construction, the resulting process satisfies :eq:`mpp`

.. _mc_eg1:

Example 1
----------

Consider a worker who, at any given time :math:`t`, is either unemployed (state 1) or employed (state 2)

Suppose that, over a one month period,

#. An unemployed worker finds a job with probability :math:`\alpha \in (0, 1)`

#. An employed worker loses her job and becomes unemployed with probability :math:`\beta \in (0, 1)`

In terms of a Markov model, we have

* :math:`S = \{ 1, 2\}`

* :math:`P(1, 2) = \alpha` and :math:`P(2, 1) = \beta`

We can write out the transition probabilities in matrix form as

.. math::

    P
    = \left(
    \begin{array}{cc}
        1 - \alpha & \alpha \\
        \beta & 1 - \beta
    \end{array}
      \right)

Once we have the values :math:`\alpha` and :math:`\beta`, we can address a range of questions, such as

* What is the average duration of unemployment?

* Over the long-run, what fraction of time does a worker find herself unemployed?

* Conditional on employment, what is the probability of becoming unemployed at least once over the next 12 months?

We'll cover such applications below

.. _mc_eg2:

Example 2
----------

Using  US unemployment data, Hamilton :cite:`Hamilton2005` estimated the stochastic matrix

.. math::

    P =
    \left(
      \begin{array}{ccc}
         0.971 & 0.029 & 0 \\
         0.145 & 0.778 & 0.077 \\
         0 & 0.508 & 0.492
      \end{array}
    \right)

where

* the frequency is monthly
* the first state represents "normal growth"
* the second state represents "mild recession"
* the third state represents "severe recession"

For example, the matrix tells us that when the state is normal growth, the state will again be normal growth next month with probability 0.97

In general, large values on the main diagonal indicate persistence in the process :math:`\{ X_t \}`

This Markov process can also be represented as a directed graph, with edges labeled by transition probabilities

.. figure:: /_static/figures/hamilton_graph.png
   :width: 80%

Here "ng" is normal growth, "mr" is mild recession, etc.

Simulation
=============

.. index::
    single: Markov Chains; Simulation

One natural way to answer questions about Markov chains is to simulate them

(To approximate the probability of event :math:`E`, we can simulate many times and count the fraction of times that :math:`E` occurs)

Nice functionality for simulating Markov chains exists in `QuantEcon.jl <http://quantecon.org/julia_index.html>`__

* Efficient, bundled with lots of other useful routines for handling Markov chains

However, it's also a good exercise to roll our own routines --- let's do that first and then come back to the methods in `QuantEcon.jl <http://quantecon.org/julia_index.html>`__

In these exercises we'll take the state space to be :math:`S = 1,\ldots, n`

Rolling our own
--------------------

To simulate a Markov chain, we need its stochastic matrix :math:`P` and either an initial state or a probability distribution :math:`\psi` for initial state to be drawn from

The Markov chain is then constructed as discussed above.  To repeat:

#. At time :math:`t=0`, the :math:`X_0` is set to some fixed state or chosen from :math:`\psi`

#. At each subsequent time :math:`t`, the new state :math:`X_{t+1}` is drawn from :math:`P(X_t, \cdot)`

In order to implement this simulation procedure, we need a method for generating draws from a discrete distributions

For this task we'll use a Categorical random variable (i.e. a discrete random variable with assigned probabilities)

.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

    d = Categorical([0.5, 0.3, 0.2]) # 3 discrete states
    @show rand(d, 5)
    @show supertype(typeof(d))
    @show pdf(d, 1) # the probability to be in state 1
    @show support(d)
    @show pdf.(d, support(d)); # broadcast the pdf over the whole support

We'll write our code as a function that takes the following three arguments

* A stochastic matrix ``P``

* An initial state ``init``

* A positive integer ``sample_size`` representing the length of the time series the function should return

.. code-block:: julia

    function mc_sample_path(P; init = 1, sample_size = 1000)
        @assert size(P)[1] == size(P)[2] # square required
        N = size(P)[1] # should be square

        # create vector of discrete RVs for each row
        dists = [Categorical(P[i, :]) for i in 1:N]

        # setup the simulation
        X = fill(0, sample_size) # allocate memory, or zeros(Int64, sample_size)
        X[1] = init # set the initial state

        for t in 2:sample_size
            dist = dists[X[t-1]] # get discrete RV from last state's transition distribution
            X[t] = rand(dist) # draw new value
        end
        return X
    end

Let's see how it works using the small matrix

.. math::
    :label: fm_smat

    P :=
    \left(
      \begin{array}{cc}
         0.4 & 0.6  \\
         0.2 & 0.8
      \end{array}
    \right)

As we'll see later, for a long series drawn from ``P``, the fraction of the sample that takes value 1 will be about 0.25

If you run the following code you should get roughly that answer

.. code-block:: julia
    :class: test

    Random.seed!(42);  # for result reproducibility

.. code-block:: julia

    P = [0.4 0.6; 0.2 0.8]
    X = mc_sample_path(P, sample_size = 100_000); # note 100_000 = 100000
    μ_1 = count(X .== 1)/length(X) # .== broadcasts test for equality. Could use mean(X .== 1)

.. code-block:: julia
    :class: test

    @testset "Sample Path Test" begin
        @test P ≈ [0.4 0.6; 0.2 0.8] # Make sure the primitive doesn't change.
        @test X[1:5] == [1, 2, 2, 1, 1]
    end

Using QuantEcon's Routines
----------------------------

As discussed above, `QuantEcon.jl <http://quantecon.org/julia_index.html>`__ has routines for handling Markov chains, including simulation

Here's an illustration using the same `P` as the preceding example

.. code-block:: julia
    :class: test

    Random.seed!(42);  # For reproducibility

.. code-block:: julia

    P = [0.4 0.6; 0.2 0.8];
    mc = MarkovChain(P)
    X = simulate(mc, 100_000);
    μ_2 = count(X .== 1)/length(X) # or mean(x -> x == 1, X)

.. code-block:: julia
    :class: test

    @testset "QE Sample Path Test" begin
        @test P ≈ [0.4 0.6; 0.2 0.8] # Make sure the primitive doesn't change.
        @test X[1:5] == [2, 2, 1, 1, 2]
        @test μ_1 ≈ μ_2 atol = 1e-4
    end

Adding state values and initial conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we wish to, we can provide a specification of state values to ``MarkovChain``

These state values can be integers, floats, or even strings

The following code illustrates

.. code-block:: julia

    mc = MarkovChain(P, ["unemployed", "employed"])
    simulate(mc, 4, init = 1) # start at state 1

.. code-block:: julia

    simulate(mc, 4, init = 2) # start at state 2

.. code-block:: julia

    simulate(mc, 4) # start with randomly chosen initial condition

.. code-block:: julia

    simulate_indices(mc, 4)

.. _mc_md:

:index:`Marginal Distributions`
===============================

.. index::
    single: Markov Chains; Marginal Distributions

Suppose that

#. :math:`\{X_t\}` is a Markov chain with stochastic matrix :math:`P`
#. the distribution of :math:`X_t` is known to be :math:`\psi_t`

What then is the distribution of :math:`X_{t+1}`, or, more generally, of :math:`X_{t+m}`?

Solution
------------

Let :math:`\psi_t` be the distribution of :math:`X_t` for :math:`t = 0, 1, 2, \ldots`

Our first aim is to find :math:`\psi_{t + 1}` given :math:`\psi_t` and :math:`P`

To begin, pick any :math:`y  \in S`

Using the `law of total probability <https://en.wikipedia.org/wiki/Law_of_total_probability>`_, we can decompose the probability that :math:`X_{t+1} = y` as follows:

.. math::

    \mathbb P \{X_{t+1} = y \}
       = \sum_{x \in S} \mathbb P \{ X_{t+1} = y \, | \, X_t = x \}
                   \cdot \mathbb P \{ X_t = x \}

In words, to get the probability of being at :math:`y` tomorrow, we account for
all  ways this can happen and sum their probabilities

Rewriting this statement in terms of  marginal and conditional probabilities gives

.. _mc_fdd:

.. math::

    \psi_{t+1}(y) = \sum_{x \in S} P(x,y) \psi_t(x)

There are :math:`n` such equations, one for each :math:`y \in S`

If we think of :math:`\psi_{t+1}` and :math:`\psi_t` as *row vectors* (as is traditional in this literature), these :math:`n` equations are summarized by the matrix expression

.. _mc_fddv:

.. math::
    :label: fin_mc_fr

    \psi_{t+1} = \psi_t P

In other words, to move the distribution forward one unit of time, we postmultiply by :math:`P`

By repeating this :math:`m` times we move forward :math:`m` steps into the future

Hence, iterating on :eq:`fin_mc_fr`, the expression :math:`\psi_{t+m} = \psi_t P^m` is also valid --- here :math:`P^m` is the :math:`m`-th power of :math:`P`

.. _mc_exfmar:

As a special case, we see that if :math:`\psi_0` is the initial distribution from
which :math:`X_0` is drawn, then :math:`\psi_0 P^m` is the distribution of
:math:`X_m`

This is very important, so let's repeat it

.. math::
    :label: mdfmc

    X_0 \sim \psi_0 \quad \implies \quad X_m \sim \psi_0 P^m

and, more generally,

.. math::
    :label: mdfmc2

    X_t \sim \psi_t \quad \implies \quad X_{t+m} \sim \psi_t P^m

.. _finite_mc_mstp:

Multiple Step Transition Probabilities
---------------------------------------

We know that the probability of transitioning from :math:`x` to :math:`y` in
one step is :math:`P(x,y)`

It turns out that the probability of transitioning from :math:`x` to :math:`y` in
:math:`m` steps is :math:`P^m(x,y)`, the :math:`(x,y)`-th element of the
:math:`m`-th power of :math:`P`

To see why, consider again :eq:`mdfmc2`, but now with :math:`\psi_t` putting all probability on state :math:`x`

* 1 in the :math:`x`-th position and zero elsewhere

Inserting this into :eq:`mdfmc2`, we see that, conditional on :math:`X_t = x`, the distribution of :math:`X_{t+m}` is the :math:`x`-th row of :math:`P^m`

In particular

.. math::

    \mathbb P \{X_{t+m} = y \} = P^m(x, y) = (x, y) \text{-th element of } P^m

Example: Probability of Recession
-----------------------------------

.. index::
    single: Markov Chains; Future Probabilities

Recall the stochastic matrix :math:`P` for recession and growth :ref:`considered above <mc_eg2>`

Suppose that the current state is unknown --- perhaps statistics are available only  at the *end* of the current month

We estimate the probability that the economy is in state :math:`x` to be :math:`\psi(x)`

The probability of being in recession (either mild or severe) in 6 months time is given by the inner product

.. math::

    \psi P^6
    \cdot
    \left(
      \begin{array}{c}
         0 \\
         1 \\
         1
      \end{array}
    \right)

.. _mc_eg1-1:

Example 2: Cross-Sectional Distributions
--------------------------------------------

.. index::
    single: Markov Chains; Cross-Sectional Distributions

The marginal distributions we have been studying can be viewed either as
probabilities or as cross-sectional frequencies in large samples

To illustrate, recall our model of employment / unemployment dynamics for a given worker :ref:`discussed above <mc_eg1>`

Consider a large (i.e., tending to infinite) population of workers, each of whose lifetime experiences are described by the specified dynamics, independently of one another

Let :math:`\psi` be the current *cross-sectional* distribution over :math:`\{ 1, 2 \}`

* For example, :math:`\psi(1)` is the unemployment rate

The cross-sectional distribution records the fractions of workers employed and unemployed at a given moment

The same distribution also describes the fractions of  a particular worker's career spent being employed and unemployed, respectively

:index:`Irreducibility and Aperiodicity`
============================================

.. index::
    single: Markov Chains; Irreducibility, Aperiodicity

Irreducibility and aperiodicity are central concepts of modern Markov chain theory

Let's see what they're about

Irreducibility
----------------

Let :math:`P` be a fixed stochastic matrix

Two states :math:`x` and :math:`y` are said to **communicate** with each other if
there exist positive integers :math:`j` and :math:`k` such that

.. math::

    P^j(x, y) > 0
    \quad \text{and} \quad
    P^k(y, x) > 0

In view of our discussion :ref:`above <finite_mc_mstp>`, this means precisely
that

* state :math:`x` can be reached eventually from state :math:`y`, and

* state :math:`y` can be reached eventually from state :math:`x`

The stochastic matrix :math:`P` is called **irreducible** if all states
communicate; that is, if :math:`x` and :math:`y` communicate for all
:math:`(x, y)` in :math:`S \times S`

For example, consider the following transition probabilities for wealth of a fictitious set of
households

.. figure:: /_static/figures/mc_irreducibility1.png
    :width: 100%

We can translate this into a stochastic matrix, putting zeros where
there's no edge between nodes

.. math::

    P :=
    \left(
      \begin{array}{ccc}
         0.9 & 0.1 & 0 \\
         0.4 & 0.4 & 0.2 \\
         0.1 & 0.1 & 0.8
      \end{array}
    \right)

It's clear from the graph that this stochastic matrix is irreducible: we can
reach any state from any other state eventually

We can also test this using `QuantEcon.jl <http://quantecon.org/julia_index.html>`__'s MarkovChain class

.. code-block:: julia

    P = [0.9 0.1 0.0; 0.4 0.4 0.2; 0.1 0.1 0.8];
    mc = MarkovChain(P)
    is_irreducible(mc)

.. code-block:: julia
    :class: test

    @testset "Irreducibility Check" begin
        @test is_irreducible(mc) == true
    end

Here's a more pessimistic scenario, where the poor are poor forever

.. figure:: /_static/figures/mc_irreducibility2.png
    :width: 100%

This stochastic matrix is not irreducible, since, for example, `rich` is not accessible from `poor`

Let's confirm this

.. code-block:: julia

    P = [1.0 0.0 0.0; 0.1 0.8 0.1; 0.0 0.2 0.8];
    mc = MarkovChain(P);
    is_irreducible(mc)

.. code-block:: julia
    :class: test

    @testset "Checking reducibility" begin
        @test !is_irreducible(mc)
    end

We can also determine the "communication classes," or the sets of communicating states (where communication refers to a nonzero probability of moving in each direction).

.. code-block:: julia

    communication_classes(mc)

It might be clear to you already that irreducibility is going to be important in terms of long run outcomes

For example, poverty is a life sentence in the second graph but not the first

We'll come back to this a bit later

Aperiodicity
----------------

Loosely speaking, a Markov chain is called periodic if it cycles in a predictible way, and aperiodic otherwise

Here's a trivial example with three states

.. figure:: /_static/figures/mc_aperiodicity1.png
    :width: 100%

The chain cycles with period 3:

.. code-block:: julia

    P = [0 1 0; 0 0 1; 1 0 0];
    mc = MarkovChain(P);
    period(mc)

.. code-block:: julia
    :class: test

    @testset "Periodicity Check" begin
        @test period(mc) == 3 # Confirm that everything is behaving as expected.
    end

More formally, the **period** of a state :math:`x` is the greatest common divisor
of the set of integers

.. math::

    D(x) := \{j \geq 1 : P^j(x, x) > 0\}

In the last example, :math:`D(x) = \{3, 6, 9, \ldots\}` for every state :math:`x`, so the period is 3

A stochastic matrix is called **aperiodic** if the period of every state is 1, and **periodic** otherwise

For example, the stochastic matrix associated with the transition probabilities below is periodic because, for example, state :math:`a` has period 2

.. figure:: /_static/figures/mc_aperiodicity2.png
    :width: 100%

We can confirm that the stochastic matrix is periodic as follows

.. code-block:: julia

    P = zeros(4, 4);
    P[1, 2] = 1;
    P[2, 1] = P[2, 3] = 0.5;
    P[3, 2] = P[3, 4] = 0.5;
    P[4, 3] = 1;
    mc = MarkovChain(P);
    period(mc)

.. code-block:: julia
    :class: test

    @testset "checking period" begin
        @test P[2,1] ≈ 0.5
        @test period(mc) == 2
    end

.. code-block:: julia

    is_aperiodic(mc)

.. code-block:: julia
    :class: test

    @testset "check if aperiodic" begin
        @test !is_aperiodic(mc)
    end

:index:`Stationary Distributions`
=================================

.. index::
    single: Markov Chains; Stationary Distributions

As seen in :eq:`fin_mc_fr`, we can shift probabilities forward one unit of time via postmultiplication by :math:`P`

Some distributions are invariant under this updating process --- for example,

.. code-block:: julia

    P = [.4 .6; .2 .8];
    ψ = [0.25, 0.75];
    ψ' * P

Such distributions are called **stationary**, or **invariant**

.. _mc_stat_dd:

Formally, a distribution :math:`\psi^*` on :math:`S` is called **stationary** for :math:`P` if :math:`\psi^* = \psi^* P`

From this equality we immediately get :math:`\psi^* = \psi^* P^t` for all :math:`t`

This tells us an important fact: If the distribution of :math:`X_0` is a stationary distribution, then :math:`X_t` will have this same distribution for all :math:`t`

Hence stationary distributions have a natural interpretation as stochastic steady states --- we'll discuss this more in just a moment

Mathematically, a stationary distribution is a fixed point of :math:`P` when :math:`P` is thought of as the map :math:`\psi \mapsto \psi P` from (row) vectors to (row) vectors

**Theorem.** Every stochastic matrix :math:`P` has at least one stationary distribution

(We are assuming here that the state space :math:`S` is finite; if not more assumptions are required)

For a proof of this result you can apply `Brouwer's fixed point theorem <https://en.wikipedia.org/wiki/Brouwer_fixed-point_theorem>`_, or see `EDTC <http://johnstachurski.net/edtc.html>`_, theorem 4.3.5

There may in fact be many stationary distributions corresponding to a given stochastic matrix :math:`P`

* For example, if :math:`P` is the identity matrix, then all distributions are stationary

Since stationary distributions are long run equilibria, to get uniqueness we require that initial conditions are not infinitely persistent

Infinite persistence of initial conditions occurs if certain regions of the
state space cannot be accessed from other regions, which is the opposite of irreducibility

This gives some intuition for the following fundamental theorem

.. _mc_conv_thm:

**Theorem.** If :math:`P` is both aperiodic and irreducible, then

#. :math:`P` has exactly one stationary distribution :math:`\psi^*`

#. For any initial distribution :math:`\psi_0`, we have :math:`\| \psi_0 P^t - \psi^* \| \to 0` as :math:`t \to \infty`

For a proof, see, for example, theorem 5.2 of :cite:`haggstrom2002finite`

(Note that part 1 of the theorem requires only irreducibility, whereas part 2
requires both irreducibility and aperiodicity)

A stochastic matrix satisfying the conditions of the theorem is sometimes called **uniformly ergodic**

One easy sufficient condition for aperiodicity and irreducibility is that every element of :math:`P` is strictly positive

* Try to convince yourself of this

Example
----------

Recall our model of employment / unemployment dynamics for a given worker :ref:`discussed above <mc_eg1>`

Assuming :math:`\alpha \in (0,1)` and :math:`\beta \in (0,1)`, the uniform ergodicity condition is satisfied

Let :math:`\psi^* = (p, 1-p)` be the stationary distribution, so that :math:`p` corresponds to unemployment (state 1)

Using :math:`\psi^* = \psi^* P` and a bit of algebra yields

.. math::

    p = \frac{\beta}{\alpha + \beta}

This is, in some sense, a steady state probability of unemployment --- more on interpretation below

Not surprisingly it tends to zero as :math:`\beta \to 0`, and to one as :math:`\alpha \to 0`

Calculating Stationary Distributions
-----------------------------------------------

.. index::
    single: Markov Chains; Calculating Stationary Distributions

As discussed above, a given Markov matrix :math:`P` can have many stationary distributions

That is, there can be many row vectors :math:`\psi` such that :math:`\psi = \psi P`

In fact if :math:`P` has two distinct stationary distributions :math:`\psi_1,
\psi_2` then it has infinitely many, since in this case, as you can verify,

.. math::

    \psi_3 := \lambda \psi_1 + (1 - \lambda) \psi_2

is a stationary distribution for :math:`P` for any :math:`\lambda \in [0, 1]`

If we restrict attention to the case where only one stationary distribution exists, one option for finding it is to try to solve the linear system :math:`\psi (I_n - P) = 0` for :math:`\psi`, where :math:`I_n` is the :math:`n \times n` identity

But the zero vector solves this equation

Hence we need to impose the restriction that the solution must be a probability distribution

A suitable algorithm is implemented in `QuantEcon.jl <http://quantecon.org/julia_index.html>`__ --- the next code block illustrates

.. code-block:: julia

    P = [.4 .6; .2 .8];
    mc = MarkovChain(P);
    stationary_distributions(mc)

The stationary distribution is unique

Convergence to Stationarity
-----------------------------------------------

.. index::
    single: Markov Chains; Convergence to Stationarity

Part 2 of the Markov chain convergence theorem :ref:`stated above <mc_conv_thm>` tells us that the distribution of :math:`X_t` converges to the stationary distribution regardless of where we start off

This adds considerable weight to our interpretation of :math:`\psi^*` as a stochastic steady state

The convergence in the theorem is illustrated in the next figure

.. code-block:: julia

    P = [0.971 0.029 0.000
         0.145 0.778 0.077
         0.000 0.508 0.492] # stochastic matrix

    ψ = [0.0 0.2 0.8] # initial distribution

    t = 20 # path length
    x_vals = zeros(t)
    y_vals = similar(x_vals)
    z_vals = similar(x_vals)
    colors = [repeat([:red], 20); :black] # for plotting

    for i in 1:t
        x_vals[i] = ψ[1]
        y_vals[i] = ψ[2]
        z_vals[i] = ψ[3]
        ψ = ψ * P # update distribution
    end

    mc = MarkovChain(P)
    ψ_star = stationary_distributions(mc)[1]
    x_star, y_star, z_star = ψ_star # unpack the stationary dist
    plt = scatter([x_vals; x_star], [y_vals; y_star], [z_vals; z_star], color = colors,
                  gridalpha = 0.5, legend = :none)
    plot!(plt, camera = (45,45))

.. code-block:: julia
    :class: test

    @testset "testing convergence to stationary" begin
        @test x_vals[12] ≈ 0.6401278316658368
        @test y_vals[7] ≈ 0.4773682392284884
    end

Here

* :math:`P` is the stochastic matrix for recession and growth :ref:`considered above <mc_eg2>`

* The highest red dot is an arbitrarily chosen initial probability distribution  :math:`\psi`, represented as a vector in :math:`\mathbb R^3`

* The other red dots are the distributions :math:`\psi P^t` for :math:`t = 1, 2, \ldots`

* The black dot is :math:`\psi^*`

The code for the figure can be found `here <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/finite_markov/mc_convergence_plot.jl>`__ --- you might like to try experimenting with different initial conditions

.. _ergodicity:

:index:`Ergodicity`
===================

.. index::
    single: Markov Chains; Ergodicity

Under irreducibility, yet another important result obtains: For all :math:`x \in S`,

.. math::
    :label: llnfmc0

    \frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = x\}  \to \psi^*(x)
        \quad \text{as } m \to \infty

Here

* :math:`\mathbf{1}\{X_t = x\} = 1` if :math:`X_t = x` and zero otherwise

* convergence is with probability one

* the result does not depend on the distribution (or value) of :math:`X_0`

The result tells us that the fraction of time the chain spends at state :math:`x` converges to :math:`\psi^*(x)` as time goes to infinity

.. _new_interp_sd:

This gives us another way to interpret the stationary distribution --- provided that the convergence result in :eq:`llnfmc0` is valid

The convergence in :eq:`llnfmc0` is a special case of a law of large numbers result for Markov chains --- see `EDTC <http://johnstachurski.net/edtc.html>`_, section 4.3.4 for some additional information

.. _mc_eg1-2:

Example
---------

Recall our cross-sectional interpretation of the employment / unemployment model :ref:`discussed above <mc_eg1-1>`

Assume that :math:`\alpha \in (0,1)` and :math:`\beta \in (0,1)`, so that irreducibility and aperiodicity both hold

We saw that the stationary distribution is :math:`(p, 1-p)`, where

.. math::

    p = \frac{\beta}{\alpha + \beta}

In the cross-sectional interpretation, this is the fraction of people unemployed

In view of our latest (ergodicity) result, it is also the fraction of time that a worker can expect to spend unemployed

Thus, in the long-run, cross-sectional averages for a population and time-series averages for a given person coincide

This is one interpretation of the notion of ergodicity

.. _finite_mc_expec:

Computing Expectations
=================================

.. index::
    single: Markov Chains; Forecasting Future Values

We are interested in computing expectations of the form

.. math::
    :label: mc_une

    \mathbb E [ h(X_t) ]

and conditional expectations such as

.. math::
    :label: mc_cce

    \mathbb E [ h(X_{t + k})  \mid X_t = x]

where

* :math:`\{X_t\}` is a Markov chain generated by :math:`n \times n` stochastic matrix :math:`P`

* :math:`h` is a given function, which, in expressions involving matrix
  algebra, we'll think of as the column vector

.. math::

    h
    = \left(
    \begin{array}{c}
        h(x_1) \\
        \vdots \\
        h(x_n) \\
    \end{array}
      \right)

The unconditional expectation :eq:`mc_une` is easy: We just sum over the
distribution of :math:`X_t` to get

.. math::

    \mathbb E [ h(X_t) ]
    = \sum_{x \in S} (\psi P^t)(x) h(x)

Here :math:`\psi` is the distribution of :math:`X_0`

Since :math:`\psi` and hence :math:`\psi P^t` are row vectors, we can also
write this as

.. math::

    \mathbb E [ h(X_t) ]
    =  \psi P^t h

For the conditional expectation :eq:`mc_cce`, we need to sum over
the conditional distribution of :math:`X_{t + k}` given :math:`X_t = x`

We already know that this is :math:`P^k(x, \cdot)`, so

.. math::
    :label: mc_cce2

    \mathbb E [ h(X_{t + k})  \mid X_t = x]
    = (P^k h)(x)

The vector :math:`P^k h` stores the conditional expectation :math:`\mathbb E [ h(X_{t + k})  \mid X_t = x]` over all :math:`x`

Expectations of Geometric Sums
------------------------------------

Sometimes we also want to compute expectations of a geometric sum, such as
:math:`\sum_t \beta^t h(X_t)`

In view of the preceding discussion, this is

.. math::

    \mathbb{E} \left[
            \sum_{j=0}^\infty \beta^j h(X_{t+j}) \mid X_t = x
        \right]
    = [(I - \beta P)^{-1} h](x)

where

.. math::

    (I - \beta P)^{-1}  = I + \beta P + \beta^2 P^2 + \cdots

Premultiplication by :math:`(I - \beta P)^{-1}` amounts to "applying the **resolvent operator**"

Exercises
==============

.. _mc_ex1:

Exercise 1
------------

According to the discussion :ref:`above <mc_eg1-2>`, if a worker's employment dynamics obey the stochastic matrix

.. math::

    P
    = \left(
    \begin{array}{cc}
        1 - \alpha & \alpha \\
        \beta & 1 - \beta
    \end{array}
      \right)

with :math:`\alpha \in (0,1)` and :math:`\beta \in (0,1)`, then, in the long-run, the fraction
of time spent unemployed will be

.. math::

    p := \frac{\beta}{\alpha + \beta}

In other words, if :math:`\{X_t\}` represents the Markov chain for
employment, then :math:`\bar X_m \to p` as :math:`m \to \infty`, where

.. math::

    \bar X_m := \frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = 1\}

Your exercise is to illustrate this convergence

First,

* generate one simulated time series :math:`\{X_t\}` of length 10,000, starting at :math:`X_0 = 1`
* plot :math:`\bar X_m - p` against :math:`m`, where :math:`p` is as defined above

Second, repeat the first step, but this time taking :math:`X_0 = 2`

In both cases, set :math:`\alpha = \beta = 0.1`

The result should look something like the following --- modulo randomness, of
course

.. figure:: /_static/figures/mc_ex1_plot.png

(You don't need to add the fancy touches to the graph --- see the solution if you're interested)

.. _mc_ex2:

Exercise 2
------------

A topic of interest for economics and many other disciplines is *ranking*

Let's now consider one of the most practical and important ranking problems
--- the rank assigned to web pages by search engines

(Although the problem is motivated from outside of economics, there is in fact a deep connection between search ranking systems and prices in certain competitive equilibria --- see :cite:`DLP2013`)

To understand the issue, consider the set of results returned by a query to a web search engine

For the user, it is desirable to

#. receive a large set of accurate matches
#. have the matches returned in order, where the order corresponds to some measure of "importance"

Ranking according to a measure of importance is the problem we now consider

The methodology developed to solve this problem by Google founders Larry Page and Sergey Brin
is known as `PageRank <https://en.wikipedia.org/wiki/PageRank>`_

To illustrate the idea, consider the following diagram

.. figure:: /_static/figures/web_graph.png
   :width: 60%

Imagine that this is a miniature version of the WWW, with

* each node representing a web page
* each arrow representing the existence of a link from one page to another

Now let's think about which pages are likely to be important, in the sense of being valuable to a search engine user

One possible criterion for importance of a page is the number of inbound links --- an indication of popularity

By this measure, ``m`` and ``j`` are the most important pages, with 5 inbound links each

However, what if the pages linking to ``m``, say, are not themselves important?

Thinking this way, it seems appropriate to weight the inbound nodes by relative importance

The PageRank algorithm does precisely this

A slightly simplified presentation that captures the basic idea is as follows

Letting :math:`j` be (the integer index of) a typical page and :math:`r_j` be its ranking, we set

.. math::

    r_j = \sum_{i \in L_j} \frac{r_i}{\ell_i}

where

* :math:`\ell_i` is the total number of outbound links from :math:`i`
* :math:`L_j` is the set of all pages :math:`i` such that :math:`i` has a link to :math:`j`

This is a measure of the number of inbound links, weighted by their own ranking (and normalized by :math:`1 / \ell_i`)

There is, however, another interpretation, and it brings us back to Markov chains

Let :math:`P` be the matrix given by :math:`P(i, j) = \mathbf 1\{i \to j\} / \ell_i` where :math:`\mathbf 1\{i \to j\} = 1` if :math:`i` has a link to :math:`j` and zero otherwise

The matrix :math:`P` is a stochastic matrix provided that each page has at least one link

With this definition of :math:`P` we have

.. math::

    r_j
    = \sum_{i \in L_j} \frac{r_i}{\ell_i}
    = \sum_{\text{all } i} \mathbf 1\{i \to j\} \frac{r_i}{\ell_i}
    = \sum_{\text{all } i} P(i, j) r_i

Writing :math:`r` for the row vector of rankings, this becomes :math:`r = r P`

Hence :math:`r` is the stationary distribution of the stochastic matrix :math:`P`

Let's think of :math:`P(i, j)` as the probability of "moving" from page :math:`i` to page :math:`j`

The value :math:`P(i, j)` has the interpretation

* :math:`P(i, j) = 1/k` if :math:`i` has :math:`k` outbound links, and :math:`j` is one of them
* :math:`P(i, j) = 0` if :math:`i` has no direct link to :math:`j`

Thus, motion from page to page is that of a web surfer who moves from one page to another by randomly clicking on one of the links on that page

Here "random" means that each link is selected with equal probability

Since :math:`r` is the stationary distribution of :math:`P`, assuming that the uniform ergodicity condition is valid, we :ref:`can interpret <new_interp_sd>` :math:`r_j` as the fraction of time that a (very persistent) random surfer spends at page :math:`j`

Your exercise is to apply this ranking algorithm to the graph pictured above,
and return the list of pages ordered by rank

When you solve for the ranking, you will find that the highest ranked node is in fact ``g``, while the lowest is ``a``

.. _mc_ex3:

Exercise 3
------------

In numerical work it is sometimes convenient to replace a continuous model with a discrete one

In particular, Markov chains are routinely generated as discrete approximations to AR(1) processes of the form

.. math::

    y_{t+1} = \rho y_t + u_{t+1}

Here :math:`{u_t}` is assumed to be i.i.d. and :math:`N(0, \sigma_u^2)`

The variance of the stationary probability distribution of :math:`\{ y_t \}` is

.. math::

    \sigma_y^2 := \frac{\sigma_u^2}{1-\rho^2}

Tauchen's method :cite:`Tauchen1986` is the most common method for approximating this continuous state process with a finite state Markov chain

A routine for this already exists in `QuantEcon.jl <http://quantecon.org/julia_index.html>`__ but let's write our own version as an exercise

As a first step we choose

* :math:`n`, the number of states for the discrete approximation
* :math:`m`, an integer that parameterizes the width of the state space

Next we create a state space :math:`\{x_0, \ldots, x_{n-1}\} \subset \mathbb R`
and a stochastic :math:`n \times n` matrix :math:`P` such that

* :math:`x_0 = - m \, \sigma_y`
* :math:`x_{n-1} = m \, \sigma_y`
* :math:`x_{i+1} = x_i + s` where :math:`s = (x_{n-1} - x_0) / (n - 1)`

Let :math:`F` be the cumulative distribution function of the normal distribution :math:`N(0, \sigma_u^2)`

The values :math:`P(x_i, x_j)` are computed to approximate the AR(1) process --- omitting the derivation, the rules are as follows:

1. If :math:`j = 0`, then set

.. math::

    P(x_i, x_j) = P(x_i, x_0) = F(x_0-\rho x_i + s/2)

2. If :math:`j = n-1`, then set

.. math::

    P(x_i, x_j) = P(x_i, x_{n-1}) = 1 - F(x_{n-1} - \rho x_i - s/2)

3. Otherwise, set

.. math::

    P(x_i, x_j) = F(x_j - \rho x_i + s/2) - F(x_j - \rho x_i - s/2)

The exercise is to write a function ``approx_markov(rho, sigma_u, m = 3, n = 7)`` that returns
:math:`\{x_0, \ldots, x_{n-1}\} \subset \mathbb R` and :math:`n \times n` matrix
:math:`P` as described above

* Even better, write a function that returns an instance of `QuantEcon.jl's <http://quantecon.org/julia_index.html>`__ `MarkovChain` type

Solutions
===========

Exercise 1
----------

Compute the fraction of time that the worker spends unemployed, and compare it
to the stationary probability.

.. code-block:: julia
    :class: test

    Random.seed!(42);  # For reproducibility

.. code-block:: julia

    α = 0.1 # probability of getting hired
    β = 0.1 # probability of getting fired
    N = 10_000
    p̄ = β / (α + β) # steady-state probabilities
    P = [1 - α   α
         β   1 - β] # stochastic matrix
    mc = MarkovChain(P)
    labels = ["start unemployed", "start employed"]
    y_vals = Array{Vector}(undef, 2) # sample paths holder

    for x0 in 1:2
        X = simulate_indices(mc, N; init = x0) # generate the sample path
        X̄ = cumsum(X .== 1) ./ (1:N) # compute state fraction. ./ required for precedence
        y_vals[x0] = X̄ .- p̄ # plot divergence from steady state
    end

    plot(y_vals, color = [:blue :green], fillrange = 0, fillalpha = 0.1,
         ylims = (-0.25, 0.25), label = reshape(labels, 1, length(labels)))

.. code-block:: julia
    :class: test

    @testset "Exercise 1 Tests" begin
        @test y_vals[2][5] ≈ -0.5
        @test X[1:5] == [2, 2, 2, 2, 2]
    end

Exercise 2
----------

.. code-block:: julia

    web_graph_data = sort(Dict('a' => ['d', 'f'],
                               'b' => ['j', 'k', 'm'],
                               'c' => ['c', 'g', 'j', 'm'],
                               'd' => ['f', 'h', 'k'],
                               'e' => ['d', 'h', 'l'],
                               'f' => ['a', 'b', 'j', 'l'],
                               'g' => ['b', 'j'],
                               'h' => ['d', 'g', 'l', 'm'],
                               'i' => ['g', 'h', 'n'],
                               'j' => ['e', 'i', 'k'],
                               'k' => ['n'],
                               'l' => ['m'],
                               'm' => ['g'],
                               'n' => ['c', 'j', 'm']))

.. code-block:: julia

    nodes = keys(web_graph_data)
    n = length(nodes)
    # create adjacency matrix of links (Q[i, j] = true for link, false otherwise)
    Q = fill(false, n, n)
    for (node, edges) in enumerate(values(web_graph_data))
        Q[node, nodes .∈ Ref(edges)] .= true
    end

    # create the corresponding stochastic matrix
    P = Q ./ sum(Q, dims = 2)

    mc = MarkovChain(P)
    r = stationary_distributions(mc)[1] # stationary distribution
    ranked_pages = Dict(zip(keys(web_graph_data), r)) # results holder

    # print solution
    println("Rankings\n ***")
    sort(collect(ranked_pages), by = x -> x[2], rev = true) # print sorted

.. code-block:: julia
    :class: test

    @testset "Exercise 2 Tests" begin
        @test ranked_pages['g'] ≈ 0.16070778858515053
        @test ranked_pages['l'] ≈ 0.032017852378295776
    end

Exercise 3
----------

A solution from `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`_ can be found `here <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/markov/markov_approx.jl>`__

.. rubric:: Footnotes

.. [#pm] Hint: First show that if :math:`P` and :math:`Q` are stochastic matrices then so is their product --- to check the row sums, try postmultiplying by a column vector of ones.  Finally, argue that :math:`P^n` is a stochastic matrix using induction.
