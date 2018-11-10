
.. _optgrowth:

.. include:: /_static/includes/lecture_howto_jl.raw

.. highlight:: julia

**************************************************************
:index:`Optimal Growth I: The Stochastic Optimal Growth Model`
**************************************************************

.. contents:: :depth: 2

Overview
============

In this lecture we're going to study a simple optimal growth model with one agent

The model is a version of the standard one sector infinite horizon growth model studied in

* :cite:`StokeyLucas1989`, chapter 2

* :cite:`Ljungqvist2012`, section 3.1

* `EDTC <http://johnstachurski.net/edtc.html>`_, chapter 1

* :cite:`Sundaram1996`, chapter 12

The technique we use to solve the model is dynamic programming

Our treatment of dynamic programming follows on from earlier
treatments in our lectures on :doc:`shortest paths <short_path>` and
:doc:`job search <mccall_model>`

We'll discuss some of the technical details of dynamic programming as we
go along

The Model
==========================

.. index::
    single: Optimal Growth; Model

Consider an agent who owns an amount :math:`y_t \in \mathbb R_+ := [0, \infty)` of a consumption good at time :math:`t`

This output can either be consumed or invested

When the good is invested it is transformed one-for-one into capital

The resulting capital stock, denoted here by :math:`k_{t+1}`, will then be used for production

Production is stochastic, in that it also depends on a shock :math:`\xi_{t+1}` realized at the end of the current period

Next period output is

.. math::

    y_{t+1} := f(k_{t+1}) \xi_{t+1}

where :math:`f \colon \RR_+ \to \RR_+` is called the production function

The resource constraint is

.. math::
    :label: outcsdp0

    k_{t+1} + c_t \leq y_t

and all variables are required to be nonnegative

Assumptions and Comments
---------------------------

In what follows,

* The sequence :math:`\{\xi_t\}` is assumed to be IID

* The common distribution of each :math:`\xi_t` will be denoted :math:`\phi`

* The production function :math:`f` is assumed to be increasing and continuous

* Depreciation of capital is not made explicit but can be incorporated into the production function

While many other treatments of the stochastic growth model use :math:`k_t` as the state variable, we will use :math:`y_t`

This will allow us to treat a stochastic model while maintaining only one state variable

We consider alternative states and timing specifications in some of our other lectures

Optimization
--------------

Taking :math:`y_0` as given, the agent wishes to maximize

.. math::
    :label: texs0_og2

    \mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(c_t) \right]

subject to

.. math::
    :label: og_conse

    y_{t+1} = f(y_t - c_t) \xi_{t+1}
    \quad \text{and} \quad
    0 \leq c_t \leq y_t
    \quad \text{for all } t

where

* :math:`u` is a bounded, continuous and strictly increasing utility function and

* :math:`\beta \in (0, 1)` is a discount factor

In :eq:`og_conse` we are assuming that the resource constraint :eq:`outcsdp0` holds with equality --- which is reasonable because :math:`u` is strictly increasing and no output will be wasted at the optimum

In summary, the agent's aim is to select a path :math:`c_0, c_1, c_2, \ldots` for consumption that is

#. nonnegative,

#. feasible in the sense of :eq:`outcsdp0`,

#. optimal, in the sense that it maximizes :eq:`texs0_og2` relative to all other feasible consumption sequences, and

#. *adapted*, in the sense that the action :math:`c_t` depends only on
   observable outcomes, not future outcomes such as :math:`\xi_{t+1}`

In the present context

* :math:`y_t` is called the *state* variable --- it summarizes the "state of the world" at the start of each period

* :math:`c_t` is called the *control* variable --- a value chosen by the agent each period after observing the state

The Policy Function Approach
--------------------------------

.. index::
    single: Optimal Growth; Policy Function Approach

One way to think about solving this problem is to look for the best **policy function**

A policy function is a map from past and present observables into current action

We'll be particularly interested in **Markov policies**, which are maps from the current state :math:`y_t` into a current action :math:`c_t`

For dynamic programming problems such as this one (in fact for any `Markov decision process <https://en.wikipedia.org/wiki/Markov_decision_process>`__), the optimal policy is always a Markov policy

In other words, the current state :math:`y_t` provides a sufficient statistic
for the history in terms of making an optimal decision today

This is quite intuitive but if you wish you can find proofs in texts such as :cite:`StokeyLucas1989` (section 4.1)

Hereafter we focus on finding the best Markov policy

In our context, a Markov policy is a function :math:`\sigma \colon
\mathbb R_+ \to \mathbb R_+`, with the understanding that states are mapped to actions via

.. math::

    c_t = \sigma(y_t) \quad \text{for all } t

In what follows, we will call :math:`\sigma` a *feasible consumption policy* if it satisfies

.. math::
    :label: idp_fp_og2

    0 \leq \sigma(y) \leq y
    \quad \text{for all} \quad
    y \in \mathbb R_+

In other words, a feasible consumption policy is a Markov policy that respects the resource constraint

The set of all feasible consumption policies will be denoted by :math:`\Sigma`

Each :math:`\sigma \in \Sigma` determines a :doc:`continuous state Markov process <stationary_densities>` :math:`\{y_t\}` for output via

.. math::
    :label: firstp0_og2

    y_{t+1} = f(y_t - \sigma(y_t)) \xi_{t+1},
    \quad y_0 \text{ given}

This is the time path for output when we choose and stick with the policy :math:`\sigma`

We insert this process into the objective function to get

.. math::
    :label: texss

    \mathbb E
    \left[ \,
    \sum_{t = 0}^{\infty} \beta^t u(c_t) \,
    \right]
    =
    \mathbb E
    \left[ \,
    \sum_{t = 0}^{\infty} \beta^t u(\sigma(y_t)) \,
    \right]

This is the total expected present value of following policy :math:`\sigma` forever,
given initial income :math:`y_0`

The aim is to select a policy that makes this number as large as possible

The next section covers these ideas more formally

Optimality
------------------------------------

The **policy value function** :math:`v_{\sigma}` associated with a given policy :math:`\sigma` is the mapping defined by

.. math::
    :label: vfcsdp00

    v_{\sigma}(y)
    =
    \mathbb E \left[ \sum_{t = 0}^{\infty} \beta^t u(\sigma(y_t)) \right]

when :math:`\{y_t\}` is given by :eq:`firstp0_og2` with :math:`y_0 = y`

In other words, it is the lifetime value of following policy :math:`\sigma`
starting at initial condition :math:`y`

The **value function** is then defined as

.. math::
    :label: vfcsdp0

    v^*(y) := \sup_{\sigma \in \Sigma} \; v_{\sigma}(y)

The value function gives the maximal value that can be obtained from state :math:`y`, after considering all feasible policies

A policy :math:`\sigma \in \Sigma` is called **optimal** if it attains the supremum in :eq:`vfcsdp0` for all :math:`y \in \mathbb R_+`

The Bellman Equation
----------------------

With our assumptions on utility and production function, the value function as defined in :eq:`vfcsdp0` also satisfies a **Bellman equation**

For this problem, the Bellman equation takes the form

.. math::
    :label: fpb30

    w(y) = \max_{0 \leq c \leq y}
        \left\{
            u(c) + \beta \int w(f(y - c) z) \phi(dz)
        \right\}
    \qquad (y \in \mathbb R_+)

This is a *functional equation in* :math:`w`

The term :math:`\int w(f(y - c) z) \phi(dz)` can be understood as the expected next period value when

* :math:`w` is used to measure value

* the state is :math:`y`

* consumption is set to :math:`c`

As shown in `EDTC <http://johnstachurski.net/edtc.html>`_, theorem 10.1.11 and a range of other texts

    *The value function* :math:`v^*` *satisfies the Bellman equation*

In other words, :eq:`fpb30` holds when :math:`w=v^*`

The intuition is that maximal value from a given state can be obtained by optimally trading off

* current reward from a given action, vs

* expected discounted future value of the state resulting from that action

The Bellman equation is important because it gives us more information about the value function

It also suggests a way of computing the value function, which we discuss below

Greedy policies
-----------------

The primary importance of the value function is that we can use it to compute optimal policies

The details are as follows

Given a continuous function :math:`w` on :math:`\mathbb R_+`, we say that :math:`\sigma \in \Sigma` is :math:`w`-**greedy** if :math:`\sigma(y)` is a solution to

.. math::
    :label: defgp20

    \max_{0 \leq c \leq y}
        \left\{
        u(c) + \beta \int w(f(y - c) z) \phi(dz)
        \right\}

for every :math:`y \in \mathbb R_+`

In other words, :math:`\sigma \in \Sigma` is :math:`w`-greedy if it optimally
trades off current and future rewards when :math:`w` is taken to be the value
function

In our setting, we have the following key result

    *A feasible consumption  policy is optimal if and only if it is* :math:`v^*`-*greedy*

The intuition is similar to the intuition for the Bellman equation, which was
provided after :eq:`fpb30`

See, for example, theorem 10.1.11 of `EDTC <http://johnstachurski.net/edtc.html>`__

Hence, once we have a good approximation to :math:`v^*`, we can compute the (approximately) optimal policy by computing the corresponding greedy policy

The advantage is that we are now solving a much lower dimensional optimization
problem

The Bellman Operator
------------------------

How, then, should we compute the value function?

One way is to use the so-called **Bellman operator**

(An operator is a map that sends functions into functions)

The Bellman operator is denoted by :math:`T` and defined by

.. math::
    :label: fcbell20_optgrowth

    Tw(y) := \max_{0 \leq c \leq y}
    \left\{
        u(c) + \beta \int w(f(y - c) z) \phi(dz)
    \right\}
    \qquad (y \in \mathbb R_+)

In other words, :math:`T` sends the function :math:`w` into the new function
:math:`Tw` defined :eq:`fcbell20_optgrowth`

By construction, the set of solutions to the Bellman equation :eq:`fpb30` *exactly coincides with* the set of fixed points of :math:`T`

For example, if :math:`Tw = w`, then, for any :math:`y \geq 0`,

.. math::

    w(y)
    = Tw(y)
    = \max_{0 \leq c \leq y}
    \left\{
        u(c) + \beta \int v^*(f(y - c) z) \phi(dz)
    \right\}

which says precisely that :math:`w` is a solution to the Bellman equation

It follows that :math:`v^*` is a fixed point of :math:`T`

Review of Theoretical Results
---------------------------------

.. index::
    single: Dynamic Programming; Theory

One can also show that :math:`T` is a contraction mapping on the set of continuous bounded functions on :math:`\mathbb R_+` under the supremum distance

.. math::

    \rho(g, h) = \sup_{y \geq 0} |g(y) - h(y)|

See  `EDTC <http://johnstachurski.net/edtc.html>`__, lemma 10.1.18

Hence it has exactly one fixed point in this set, which we know is equal to the value function

It follows that

* The value function :math:`v^*` is bounded and continuous

* Starting from any bounded and continuous :math:`w`, the sequence :math:`w, Tw, T^2 w, \ldots` generated by iteratively applying :math:`T` converges uniformly to :math:`v^*`

This iterative method is called **value function iteration**

We also know that a feasible policy is optimal if and only if it is :math:`v^*`-greedy

It's not too hard to show that a :math:`v^*`-greedy policy exists (see  `EDTC <http://johnstachurski.net/edtc.html>`__, theorem 10.1.11 if you get stuck)

Hence at least one optimal policy exists

Our problem now is how to compute it

:index:`Unbounded Utility`
--------------------------

.. index::
    single: Dynamic Programming; Unbounded Utility

The results stated above assume that the utility function is bounded

In practice economists often work with unbounded utility functions --- and so will we

In the unbounded setting, various optimality theories exist

Unfortunately, they tend to be case specific, as opposed to valid for a large range of applications

Nevertheless, their main conclusions are usually in line with those stated for
the bounded case just above (as long as we drop the word "bounded")

Consult,  for example, section 12.2 of `EDTC <http://johnstachurski.net/edtc.html>`_, :cite:`Kamihigashi2012` or :cite:`MV2010`

Computation
=============

.. index::
    single: Dynamic Programming; Computation

Let's now look at computing the value function and the optimal policy

Fitted Value Iteration
-------------------------

.. index::
    single: Dynamic Programming; Value Function Iteration

The first step is to compute the value function by value function iteration

In theory, the algorithm is as follows

#. Begin with a function :math:`w` --- an initial condition

#. Solving :eq:`fcbell20_optgrowth`, obtain the function :math:`T w`

#. Unless some stopping condition is satisfied, set :math:`w = Tw` and go to step 2

This generates the sequence :math:`w, Tw, T^2 w, \ldots`

However, there is a problem we must confront before we implement this procedure: The iterates can neither be calculated exactly nor stored on a computer

To see the issue, consider :eq:`fcbell20_optgrowth`

Even if :math:`w` is a known function, unless :math:`Tw` can be shown to have
some special structure, the only way to store it is to record the
value :math:`Tw(y)` for every :math:`y \in \mathbb R_+`

Clearly this is impossible

What we will do instead is use **fitted value function iteration**

The procedure is to record the value of the function :math:`Tw` at only finitely many "grid" points :math:`y_1 < y_2 < \cdots < y_I` and reconstruct it from this information when required

More precisely, the algorithm will be

.. _fvi_alg:

#. Begin with an array of values :math:`\{ w_1, \ldots, w_I \}` representing the values of some initial function :math:`w` on the grid points :math:`\{ y_1, \ldots, y_I \}`

#. Build a function :math:`\hat w` on the state space :math:`\mathbb R_+` by interpolation or approximation, based on these data points

#.  Obtain and record the value :math:`T \hat w(y_i)` on each grid point :math:`y_i` by repeatedly solving :eq:`fcbell20_optgrowth`

#. Unless some stopping condition is satisfied, set :math:`\{ w_1, \ldots, w_I \} = \{ T \hat w(y_1), \ldots, T \hat w(y_I) \}` and go to step 2

How should we go about step 2?

This is a problem of function approximation, and there are many ways to approach it

What's important here is that the function approximation scheme must not only produce a good approximation to :math:`Tw`, but also combine well with the broader iteration algorithm described above

.. only:: html

    One good choice from both respects is continuous piecewise linear interpolation (see :download:`this paper </_static/pdfs/3ndp.pdf>` for further discussion)

.. only:: latex

    One good choice from both respects is continuous piecewise linear interpolation (see `this paper <https://lectures.quantecon.org/_downloads/3ndp.pdf>`__ for further discussion)

The next figure illustrates piecewise linear interpolation of an arbitrary function on grid points :math:`0, 0.2, 0.4, 0.6, 0.8, 1`

Setup
------------------

.. literalinclude:: /_static/includes/deps.jl

.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

    using Plots, QuantEcon, Interpolations, NLsolve
    gr(fmt = :png)

.. code-block:: julia

    f(x) = 2 .* cos.(6x) .+ sin.(14x) .+ 2.5
    c_grid = 0:.2:1
    f_grid = range(0,  1, length = 150)

    Af = LinearInterpolation(c_grid, f(c_grid))

    plt = plot(xlim = (0,1), ylim = (0,6))
    plot!(plt, f, f_grid, color = :blue, lw = 2, alpha = 0.8, label = "true function")
    plot!(plt, f_grid, Af.(f_grid), color = :green, lw = 2, alpha = 0.8,
          label = "linear approximation")
    plot!(plt, f, c_grid, seriestype = :sticks, linestyle = :dash, linewidth = 2, alpha = 0.5, 
          label = "")
    plot!(plt, legend = :top)

Another advantage of piecewise linear interpolation is that it preserves useful shape properties such as monotonicity and concavity / convexity

The Bellman Operator
-----------------------

Here's a function that implements the Bellman operator using linear interpolation

.. code-block:: julia

    using Optim

    function bellman_operator(w, grid, β, u, f, shocks, Tw = similar(w); 
                              compute_policy = false)
        w_func = LinearInterpolation(grid, w)
        # objective for each grid point
        objectives = (c -> u(c) + β * mean(w_func.(f(y - c) .* shocks)) for y in grid_y) 
        results = maximize.(objectives, 1e-10, grid_y) # solver result for each grid point
        Tw = Optim.maximum.(results)
        if compute_policy
            σ = Optim.maximizer.(results)
            return Tw, σ
        end
        return Tw
    end

Notice that the expectation in :eq:`fcbell20_optgrowth` is computed via Monte Carlo, using the approximation

.. math::

    \int w(f(y - c) z) \phi(dz) \approx \frac{1}{n} \sum_{i=1}^n w(f(y - c) \xi_i)

where :math:`\{\xi_i\}_{i=1}^n` are IID draws from :math:`\phi`

Monte Carlo is not always the most efficient way to compute integrals numerically but it does have some theoretical advantages in the present setting

(For example, it preserves the contraction mapping property of the Bellman operator --- see, e.g., :cite:`pal2013`)

.. _benchmark_growth_mod:

An Example
-----------------------------

Let's test out our operator when

* :math:`f(k) = k^{\alpha}`

* :math:`u(c) = \ln c`

* :math:`\phi` is the distribution of :math:`\exp(\mu + \sigma \zeta)` when :math:`\zeta` is standard normal

As is well-known (see :cite:`Ljungqvist2012`, section 3.1.2), for this particular problem an exact analytical solution is available, with

.. math::
    :label: dpi_tv

    v^*(y) =
    \frac{\ln (1 - \alpha \beta) }{ 1 - \beta}
    +
    \frac{(\mu + \alpha \ln (\alpha \beta))}{1 - \alpha}
     \left[
         \frac{1}{1- \beta} - \frac{1}{1 - \alpha \beta}
     \right]
     +
     \frac{1}{1 - \alpha \beta} \ln y

The optimal consumption policy is

.. math::

    \sigma^*(y) = (1 - \alpha \beta ) y

Let's code this up now so we can test against it below

.. code-block:: julia

    α = 0.4
    β = 0.96
    μ = 0
    s = 0.1

    c1 = log(1 - α * β) / (1 - β)
    c2 = (μ + α * log(α * β)) / (1 - α)
    c3 = 1 / (1 - β)
    c4 = 1 / (1 - α * β)

    # Utility
    u(c) = log(c)

    ∂u∂c(c) = 1 / c

    # Deterministic part of production function
    f(k) = k^α

    f′(k) = α * k^(α - 1)

    # True optimal policy
    c_star(y) = (1 - α * β) * y

    # True value function
    v_star(y) = c1 + c2 * (c3 - c4) + c4 * log(y)

.. code-block:: julia
    :class: test

    @testset "Primitives Tests" begin
        @test [c1, c2, c3, c4] ≈ [-12.112707886215421, -0.6380751509296068, 24.99999999999998,  
                                  1.6233766233766234]
        @test ∂u∂c(c1) ≈ -0.08255792258789846
        @test v_star(3) ≈ -25.245288867900843
    end

A First Test
--------------

To test our code, we want to see if we can replicate the analytical solution numerically, using fitted value function iteration

We need a grid and some shock draws for Monte Carlo integration

.. code-block:: julia

    using Random
    Random.seed!(42) # For reproducible results.

    grid_max = 4         # Largest grid point
    grid_size = 200      # Number of grid points
    shock_size = 250     # Number of shock draws in Monte Carlo integral

    grid_y = range(1e-5,  grid_max, length = grid_size)
    shocks = exp.(μ .+ s * randn(shock_size))

.. code-block:: julia
    :class: test

    @testset "Shock Invariance Tests" begin
        @test shocks[4] ≈ 0.9704956010607036 && length(shocks) == 250
    end

Now let's do some tests

As one preliminary test, let's see what happens when we apply our Bellman operator to the exact solution :math:`v^*`

In theory, the resulting function should again be :math:`v^*`

In practice we expect some small numerical error

.. code-block:: julia

  w = bellman_operator(v_star.(grid_y), grid_y, β, log, k -> k^α, shocks)

  plt = plot(ylim = (-35,-24))
  plot!(plt, grid_y, w, linewidth = 2, alpha = 0.6, label = "bellman_operator(v_star)")
  plot!(plt, v_star, grid_y, linewidth = 2, alpha=0.6, label = "v_star")
  plot!(plt, legend = :bottomright)

.. code-block:: julia
    :class: test

    @testset "Bellman Operator Tests" begin
        @test w[4] ≈ -31.59897775567377
        @test length(w) == 200
    end

The two functions are essentially indistinguishable, so we are off to a good start

Now let's have a look at iterating with the Bellman operator, starting off
from an arbitrary initial condition

The initial condition we'll start with is :math:`w(y) = 5 \ln (y)`

.. code-block:: julia

    w = 5 * log.(grid_y)  # An initial condition -- fairly arbitrary
    n = 35

    plot(xlim = (extrema(grid_y)), ylim = (-50, 10))
    lb = "initial condition"
    plt = plot(grid_y, w, color = :black, linewidth = 2, alpha = 0.8, label = lb)
    for i in 1:n
        w = bellman_operator(w, grid_y, β, log, k -> k^α, shocks)
        plot!(grid_y, w, color = RGBA(i/n, 0, 1 - i/n, 0.8), linewidth = 2, alpha = 0.6, 
              label = "")
    end

    lb = "true value function"
    plot!(plt, v_star, grid_y, color = :black, linewidth = 2, alpha = 0.8, label = lb)
    plot!(plt, legend = :bottomright)

.. code-block:: julia
    :class: test

    @testset begin
        @test v_star(grid_y[2]) ≈ -33.370496456772266
    end

The figure shows

#. the first 36 functions generated by the fitted value function iteration algorithm, with hotter colors given to higher iterates

#. the true value function :math:`v^*` drawn in black

The sequence of iterates converges towards :math:`v^*`

We are clearly getting closer

We can write a function that computes the exact fixed point

.. code-block:: julia

    function solve_optgrowth(initial_w; tol = 1e-6, max_iter = 500)
        Tw = similar(grid_y)
        v_star_approx = fixedpoint(w -> bellman_operator(w, grid_y, β, u, f, shocks, Tw), 
                                   initial_w).zero # gets returned
    end

We can check our result by plotting it against the true value

.. code-block:: julia

    initial_w = 5 * log.(grid_y)
    v_star_approx = solve_optgrowth(initial_w)

    plt = plot(ylim = (-35, -24))
    plot!(plt, grid_y, v_star_approx, linewidth = 2, alpha = 0.6, 
          label = "approximate value function")
    plot!(plt, v_star, grid_y, linewidth = 2, alpha = 0.6, label = "true value function")
    plot!(plt, legend = :bottomright)

.. code-block:: julia
    :class: test

    @testset "Iteration Scheme Tests" begin
        @test v_star_approx[4] ≈ -31.85032776321786
    end

The figure shows that we are pretty much on the money

The Policy Function
-----------------------

.. index::
    single: Optimal Growth; Policy Function

To compute an approximate optimal policy, we take the approximate value
function we just calculated and then compute the corresponding greedy policy

The next figure compares the result to the exact solution, which, as mentioned
above, is :math:`\sigma(y) = (1 - \alpha \beta) y`

.. code-block:: julia

    Tw, σ = bellman_operator(v_star_approx, grid_y, β, log, k -> k^α, shocks; 
                             compute_policy = true)
    cstar = (1 - α * β) * grid_y

    plt = plot(grid_y, σ, lw=2, alpha=0.6, label = "approximate policy function")
    plot!(plt, grid_y, cstar, lw = 2, alpha = 0.6, label = "true policy function")
    plot!(plt, legend = :bottomright)

.. code-block:: julia
    :class: test

    @testset begin
        @test cstar[102] == 1.2505758978894472
    end

The figure shows that we've done a good job in this instance of approximating
the true policy

Exercises
=============

Exercise 1
------------

Once an optimal consumption policy :math:`\sigma` is given, income follows :eq:`firstp0_og2`

The next figure shows a simulation of 100 elements of this sequence for three different discount factors (and hence three different policies)

.. figure:: /_static/figures/solution_og_ex2.png
   :scale: 100%

In each sequence, the initial condition is :math:`y_0 = 0.1`

The discount factors are ``discount_factors = (0.8, 0.9, 0.98)``

We have also dialed down the shocks a bit

.. code-block:: julia
    :class: test

    Random.seed!(42);

.. code-block:: julia

    s = 0.05
    shocks = exp.(μ .+ s * randn(shock_size))

.. code-block:: julia
    :class: test

    @testset begin
        @test shocks[25] == 0.8050318706532391
    end

Otherwise, the parameters and primitives are the same as the log linear model discussed earlier in the lecture

Notice that more patient agents typically have higher wealth

Replicate the figure modulo randomness

Solutions
==========

Exercise 1
---------------------------

Here's one solution (assuming as usual that you've executed everything above)

.. code-block:: julia

    function simulate_og(σ, y0 = 0.1, ts_length=100)
        y = zeros(ts_length)
        ξ = randn(ts_length-1)
        y[1] = y0
        for t in 1:(ts_length-1)
            y[t+1] = (y[t] - σ(y[t]))^α * exp(μ + s * ξ[t])
        end
        return y
    end

    plt = plot()

    for β in (0.9, 0.94, 0.98)
        Tw = similar(grid_y)
        initial_w = 5 * log.(grid_y)

        v_star_approx = fixedpoint(w -> bellman_operator(w, grid_y, β, u, f, shocks, Tw),                                  initial_w).zero
        Tw, σ = bellman_operator(v_star_approx, grid_y, β, log, k -> k^α, shocks,
                                 compute_policy = true)
        σ_func = LinearInterpolation(grid_y, σ)
        y = simulate_og(σ_func)

        plot!(plt, y, lw = 2, alpha = 0.6, label = label = "beta = $β")
    end

    plot!(plt, legend = :bottomright)

.. code-block:: julia
    :class: test

    using Random
    Random.seed!(42)
    β = 0.94

    Tw = similar(grid_y)
    initial_w = 5 * log.(grid_y)
    v_star_approx = fixedpoint(w -> bellman_operator(w, grid_y, β, u, f, shocks, Tw), 
                               initial_w).zero
    Tw, σ = bellman_operator(v_star_approx, grid_y, β, log, k -> k^α, shocks, 
                             compute_policy = true)
    σ_func = LinearInterpolation(grid_y, σ)
    y = simulate_og(σ_func);

    @testset begin
        @test y[5] ≈ 0.489631315122766
        @test σ[3] ≈ 0.025086037159168094
        @test Tw[4] ≈ -22.27089960340596
        @test v_star_approx[50] ≈ -17.79360247626825
    end
