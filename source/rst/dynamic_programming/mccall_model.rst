.. _mccall:

.. include:: /_static/includes/header.raw

.. highlight:: julia

******************************************
Job Search I: The McCall Search Model
******************************************

.. contents:: :depth: 2

.. epigraph::

    "Questioning a McCall worker is like having a conversation with an out-of-work friend:
    'Maybe you are setting your sights too high', or 'Why did you quit your old job before you
    had a new one lined up?' This is real social science: an attempt to model, to understand,
    human behavior by visualizing the situation people find themselves in, the options they face
    and the pros and cons as they themselves see them." -- Robert E. Lucas, Jr.

Overview
============

The McCall search model :cite:`McCall1970` helped transform economists' way of thinking about labor markets

To clarify vague notions such as "involuntary" unemployment, McCall modeled the decision problem of unemployed agents directly, in terms of factors such as

*  current and likely future wages

*  impatience

*  unemployment compensation

To solve the decision problem he used dynamic programming

Here we set up McCall's model and adopt the same solution method

As we'll see, McCall's model is not only interesting in its own right but also an excellent vehicle for learning dynamic programming

The McCall Model
=================

.. index::
    single: Models; McCall

An unemployed worker receives in each period a job offer at wage :math:`W_t`

At time :math:`t`, our worker has two choices:

#. Accept the offer and work permanently at constant wage :math:`W_t`

#. Reject the offer, receive unemployment compensation :math:`c`, and reconsider next period

The wage sequence :math:`\{W_t\}` is assumed to be iid with probability mass function :math:`p_1, \ldots, p_n`

Here :math:`p_i` is the probability of observing wage offer :math:`W_t = w_i` in the set :math:`w_1, \ldots, w_n`

The worker is infinitely lived and aims to maximize the expected discounted sum of earnings

.. math::
    \mathbb{E} \sum_{t=0}^{\infty} \beta^t Y_t

The constant :math:`\beta` lies in :math:`(0, 1)` and is called a **discount factor**

The smaller is :math:`\beta`, the more the worker discounts future utility relative to current utility

The variable  :math:`Y_t` is income, equal to

* his wage :math:`W_t` when employed

* unemployment compensation :math:`c` when unemployed

A Trade Off
--------------------

The worker faces a trade-off:

* Waiting too long for a good offer is costly, since the future is discounted

* Accepting too early is costly, since better offers might arrive in the future

To decide optimally in the face of this trade off, we use dynamic programming

Dynamic programming can be thought of as a two step procedure that

#. first assigns values to "states" and

#. then deduces optimal actions given those values

We'll go through these steps in turn

The Value Function
---------------------

In order to optimally trade off current and future rewards, we need to think about two things:

#. the current payoffs we get from different choices

#. the different states that those choices will lead to next period (in this case, either employment or unemployment)

To weigh these two aspects of the decision problem, we need to assign *values* to states

To this end, let :math:`V(w)` be the total lifetime *value* accruing to an unemployed worker who enters the current period unemployed but with wage offer :math:`w` in hand

More precisely, :math:`V(w)` denotes the value of the objective function :eq:`objective` when an agent in this situation makes *optimal* decisions now and at all future points in time

Of course :math:`V(w)` is not trivial to calculate because we don't yet know what decisions are optimal and what aren't!

But think of :math:`V` as a function that assigns to each possible wage :math:`w` the maximal lifetime value that can be obtained with that offer in hand

A crucial observation is that this function :math:`V` must satisfy the recursion

.. math::
    :label: odu_pv

    V(w)
    = \max \left\{
            \frac{w}{1 - \beta}, \, c + \beta \sum_{i=1}^n V(w_i) p_i
        \right\}

for every possible :math:`w_i`  in :math:`w_1, \ldots, w_n`

This important equation is a version of the **Bellman equation**, which is
ubiquitous in economic dynamics and other fields involving planning over time

The intuition behind it is as follows:

* the first term inside the max operation is the lifetime payoff from accepting current offer :math:`w`, since

.. math::
    w + \beta w + \beta^2 w + \cdots = \frac{w}{1 - \beta}

* the second term inside the max operation is the **continuation value**, which is the lifetime payoff from rejecting the current offer and then behaving optimally in all subsequent periods

If we optimize and pick the best of these two options, we obtain maximal lifetime value from today, given current offer :math:`w`

But this is precisely :math:`V(w)`, which is the l.h.s. of :eq:`odu_pv`

The Optimal Policy
-------------------

Suppose for now that we are able to solve :eq:`odu_pv` for the unknown
function :math:`V`

Once we have this function in hand we can behave optimally (i.e., make the
right choice between accept and reject)

All we have to do is select the maximal choice on the r.h.s. of :eq:`odu_pv`

The optimal action is best thought of as a **policy**, which is, in general, a map from
states to actions

In our case, the state is the current wage offer :math:`w`

Given *any* :math:`w`, we can read off the corresponding best choice (accept or
reject) by picking the max on the r.h.s. of :eq:`odu_pv`

Thus, we have a map from :math:`\mathbb{R}` to :math:`\{0, 1\}`, with 1 meaning accept and zero meaning reject

We can write the policy as follows

.. math::
    \sigma(w) := \mathbf{1}
        \left\{
            \frac{w}{1 - \beta} \geq c + \beta \sum_{i=1}^n V(w_i) p_i
        \right\}

Here :math:`\mathbf{1}\{ P \} = 1` if statement :math:`P` is true and equals zero otherwise

We can also write this as

.. math::
    \sigma(w) := \mathbf{1} \{ w \geq \bar w \}

where

.. math::
    :label: odu_barw

    \bar w := (1 - \beta) \left\{ c + \beta \sum_{i=1}^n V(w_i) p_i \right\}

Here :math:`\bar w` is a constant depending on :math:`\beta, c` and the wage distribution, called the *reservation wage*

The agent should accept if and only if the current wage offer exceeds the reservation wage

Clearly, we can compute this reservation wage if we can compute the value function

Computing the Optimal Policy: Take 1
======================================

To put the above ideas into action, we need to compute the value function at
points :math:`w_1, \ldots, w_n`

In doing so, we can identify these values with the vector :math:`v = (v_i)` where :math:`v_i := V(w_i)`

In view of :eq:`odu_pv`, this vector satisfies the nonlinear system of equations

.. math::
    :label: odu_pv2

    v_i
    = \max \left\{
            \frac{w_i}{1 - \beta}, \, c + \beta \sum_{i=1}^n v_i p_i
        \right\}
    \quad
    \text{for } i = 1, \ldots, n

It turns out that there is exactly one vector :math:`v := (v_i)_{i=1}^n` in
:math:`\mathbb R^n` that satisfies this equation

The Algorithm
-------------

To compute this vector, we proceed as follows:

Step 1: pick an arbitrary initial guess :math:`v \in \mathbb R^n`

Step 2: compute a new vector :math:`v' \in \mathbb R^n` via

.. math::
    :label: odu_pv2p

    v'_i
    = \max \left\{
            \frac{w_i}{1 - \beta}, \, c + \beta \sum_{i=1}^n v_i p_i
        \right\}
    \quad
    \text{for } i = 1, \ldots, n

Step 3: calculate a measure of the deviation between :math:`v` and :math:`v'`, such as :math:`\max_i |v_i - v_i'|`

Step 4: if the deviation is larger than some fixed tolerance, set :math:`v = v'` and go to step 2, else continue

Step 5: return :math:`v`

This algorithm returns an arbitrarily good approximation to the true solution
to :eq:`odu_pv2`, which represents the value function

(Arbitrarily good means here that the approximation converges to the true
solution as the tolerance goes to zero)

The Fixed Point Theory
-----------------------

What's the math behind these ideas?

First, one defines a mapping :math:`T` from :math:`\mathbb R^n` to
itself via

.. math::
    :label: odu_pv3

    Tv_i
    = \max \left\{
            \frac{w_i}{1 - \beta}, \, c + \beta \sum_{i=1}^n v_i p_i
        \right\}
    \quad
    \text{for } i = 1, \ldots, n

(A new vector :math:`Tv` is obtained from given vector :math:`v` by evaluating
the r.h.s. at each :math:`i`)

One can show that the conditions of the Banach contraction mapping theorem are
satisfied by :math:`T` as a self-mapping on :math:`\mathbb{R}^n`

One implication is that :math:`T` has a unique fixed point in :math:`\mathbb R^n`

Moreover, it's immediate from the definition of :math:`T` that this fixed
point is precisely the value function

The iterative algorithm presented above corresponds to iterating with
:math:`T` from some initial guess :math:`v`

The Banach contraction mapping theorem tells us that this iterative process
generates a sequence that converges to the fixed point

Implementation
----------------

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics
    using Distributions, Expectations, NLsolve, Roots, Random, Plots, Parameters

.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

    gr(fmt = :png);;

Here's the distribution of wage offers we'll work with

.. code-block:: julia

    n = 50
    dist = BetaBinomial(n, 200, 100) # probability distribution
    @show support(dist)
    w = range(10.0, 60.0, length = n+1) # linearly space wages

    using StatsPlots
    plt = plot(w, dist, xlabel = "wages", ylabel = "probabilities", legend = false)

We can explore taking expectations over this distribution

.. code-block:: julia

    E = expectation(dist) # expectation operator

    # exploring the properties of the operator
    wage(i) = w[i+1] # +1 to map from support of 0
    E_w = E(wage)
    E_w_2 = E(i -> wage(i)^2) - E_w^2 # variance
    @show E_w, E_w_2

    # use operator with left-multiply
    @show E * w # the `w` are values assigned for the discrete states
    @show dot(pdf.(dist, support(dist)), w); # identical calculation


To implement our algorithm, let's have a look at the sequence of approximate value functions that
this fixed point algorithm generates

Default parameter values are embedded in the function

Our initial guess :math:`v` is the value of accepting at every given wage

.. code-block:: julia

    # parameters and constant objects

    c = 25
    β = 0.99
    num_plots = 6

    # Operator
    T(v) = max.(w/(1 - β), c + β * E*v) # (5) broadcasts over the w, fixes the v
    # alternatively, T(v) = [max(wval/(1 - β), c + β * E*v) for wval in w]

    # fill in  matrix of vs
    vs = zeros(n + 1, 6) # data to fill
    vs[:, 1] .= w / (1-β) # initial guess of "accept all"

    # manually applying operator
    for col in 2:num_plots
        v_last = vs[:, col - 1]
        vs[:, col] .= T(v_last)  # apply operator
    end
    plot(vs)

One approach to solving the model is to directly implement this sort of iteration, and continues until measured deviation
between successive iterates is below `tol`

.. code-block:: julia

    function compute_reservation_wage_direct(params; v_iv = collect(w ./(1-β)), max_iter = 500,
                                             tol = 1e-6)
        @unpack c, β, w = params

        # create a closure for the T operator
        T(v) = max.(w/(1 - β), c + β * E*v) # (5) fixing the parameter values

        v = copy(v_iv) # start at initial value.  copy to prevent v_iv modification
        v_next = similar(v)
        i = 0
        error = Inf
        while i < max_iter && error > tol
            v_next .= T(v) # (4)
            error = norm(v_next - v)
            i += 1
            v .= v_next  # copy contents into v.  Also could have used v[:] = v_next
        end
        # now compute the reservation wage
        return (1 - β) * (c + β * E*v) # (2)
    end

In the above, we use ``v = copy(v_iv)`` rather than just ``v_iv = v``

To understand why, first recall that ``v_iv`` is a function argument -- either defaulting to the given value, or passed into the function

  * If we had gone ``v = v_iv`` instead, then it would have simply created a new name ``v`` which binds to whatever is located at ``v_iv``
  * Since we later use ``v .= v_next`` later in the algorithm, the values in it would be modified
  * Hence, we would be modifying the ``v_iv`` vector we were passed in, which may not be what the caller of the function wanted
  * The big issue this creates are "side-effects" where you can call a function and strange things can happen outside of the function that you didn't expect
  * If you intended for the modification to potentially occur, then the Julia style guide says that we should call the function ``compute_reservation_wage_direct!`` to make the possible side-effects clear


As usual, we are better off using a package, which may give a better algorithm and is likely to less error prone

In this case, we can use the ``fixedpoint`` algorithm discussed in :doc:`our Julia by Example lecture <../getting_started_julia/julia_by_example>`  to find the fixed point of the :math:`T` operator

.. code-block:: julia

    function compute_reservation_wage(params; v_iv = collect(w ./(1-β)), iterations = 500,
                                      ftol = 1e-6, m = 6)
        @unpack c, β, w = params
        T(v) = max.(w/(1 - β), c + β * E*v) # (5) fixing the parameter values

        v_star = fixedpoint(T, v_iv, iterations = iterations, ftol = ftol,
                            m = 6).zero # (5)
        return (1 - β) * (c + β * E*v_star) # (3)
    end

Let's compute the reservation wage at the default parameters

.. code-block:: julia

    mcm = @with_kw (c=25.0, β=0.99, w=w) # named tuples

    compute_reservation_wage(mcm()) # call with default parameters

.. code-block:: julia
    :class: test

    @testset "Reservation Wage Tests" begin
        @test compute_reservation_wage(mcm()) ≈ 47.316499766546215
        @test compute_reservation_wage_direct(mcm()) ≈ 47.31649975736077
    end

Comparative Statics
-------------------

Now we know how to compute the reservation wage, let's see how it varies with
parameters

In particular, let's look at what happens when we change :math:`\beta` and
:math:`c`

.. code:: julia

    grid_size = 25
    R = rand(grid_size, grid_size)

    c_vals = range(10.0, 30.0, length = grid_size)
    β_vals = range(0.9, 0.99, length = grid_size)

    for (i, c) in enumerate(c_vals)
        for (j, β) in enumerate(β_vals)
            R[i, j] = compute_reservation_wage(mcm(c=c, β=β)) # change from defaults
        end
    end

.. code-block:: julia
    :class: test

    @testset "Comparative Statics Tests" begin
        @test R[4, 4] ≈ 41.15851842606614 # arbitrary reservation wage.
        @test grid_size == 25 # grid invariance.
        @test length(c_vals) == grid_size && c_vals[1] ≈ 10.0 && c_vals[end] ≈ 30.0
        @test length(β_vals) == grid_size && β_vals[1] ≈ 0.9 && β_vals[end] ≈ 0.99
    end

.. code-block:: julia

    contour(c_vals, β_vals, R',
            title = "Reservation Wage",
            xlabel = "c",
            ylabel = "beta",
            fill = true)

As expected, the reservation wage increases both with patience and with
unemployment compensation

Computing the Optimal Policy: Take 2
======================================

The approach to dynamic programming just described is very standard and
broadly applicable

For this particular problem, there's also an easier way, which circumvents the
need to compute the value function

Let :math:`\psi` denote the value of not accepting a job in this period but
then behaving optimally in all subsequent periods

That is,

.. math::
    :label: j1

    \psi
    = c + \beta
        \sum_{i=1}^n V(w_i) p_i

where :math:`V` is the value function

By the Bellman equation, we then have

.. math::

    V(w_i)
    = \max \left\{ \frac{w_i}{1 - \beta}, \, \psi \right\}

Substituting this last equation into :eq:`j1` gives

.. math::
    :label: j2

    \psi
    = c + \beta
        \sum_{i=1}^n
        \max \left\{
            \frac{w_i}{1 - \beta}, \psi
        \right\}  p_i

Which we could also write as :math:`\psi = T_{\psi}(\psi)` for the appropriate operator

This is a nonlinear equation that we can solve for :math:`\psi`

One solution method for this kind of nonlinear equation is iterative

That is,

Step 1: pick an initial guess :math:`\psi`

Step 2: compute the update :math:`\psi'` via

.. math::
    :label: j3

    \psi'
    = c + \beta
        \sum_{i=1}^n
        \max \left\{
            \frac{w_i}{1 - \beta}, \psi
        \right\}  p_i

Step 3: calculate the deviation :math:`|\psi - \psi'|`

Step 4: if the deviation is larger than some fixed tolerance, set :math:`\psi = \psi'` and go to step 2, else continue

Step 5: return :math:`\psi`

Once again, one can use the Banach contraction mapping theorem to show that this process always converges

The big difference here, however, is that we're iterating on a single number, rather than an :math:`n`-vector

Here's an implementation:

.. code-block:: julia

    function compute_reservation_wage_ψ(c, β; ψ_iv = E * w ./ (1 - β), max_iter = 500,
                                        tol = 1e-5)
        T_ψ(ψ) = [c + β * E*max.((w ./ (1 - β)), ψ[1])] # (7)
        # using vectors since fixedpoint doesn't support scalar
        ψ_star = fixedpoint(T_ψ, [ψ_iv]).zero[1]
        return (1 - β) * ψ_star # (2)
    end
    compute_reservation_wage_ψ(c, β)

You can use this code to solve the exercise below

Another option is to solve for the root of the  :math:`T_{\psi}(\psi) - \psi` equation

.. code-block:: julia

    function compute_reservation_wage_ψ2(c, β; ψ_iv = E * w ./ (1 - β), max_iter = 500,
                                         tol = 1e-5)
        root_ψ(ψ) = c + β * E*max.((w ./ (1 - β)), ψ) - ψ # (7)
        ψ_star = find_zero(root_ψ, ψ_iv)
        return (1 - β) * ψ_star # (2)
    end
    compute_reservation_wage_ψ2(c, β)

.. code-block:: julia
    :class: test

    @testset begin
        mcmp = mcm()
        @test compute_reservation_wage(mcmp) ≈ 47.316499766546215
        @test compute_reservation_wage_ψ(mcmp.c, mcmp.β) ≈ 47.31649976654623
        @test compute_reservation_wage_ψ2(mcmp.c, mcmp.β) ≈ 47.31649976654623
    end
    
Exercises
=========

Exercise 1
------------

Compute the average duration of unemployment when :math:`\beta=0.99` and
:math:`c` takes the following values

    ``c_vals = range(10, 40, length = 25)``

That is, start the agent off as unemployed, computed their reservation wage
given the parameters, and then simulate to see how long it takes to accept

Repeat a large number of times and take the average

Plot mean unemployment duration as a function of :math:`c` in ``c_vals``

Solutions
==========

Exercise 1
----------

Here's one solution

.. code:: julia

    function compute_stopping_time(w̄; seed=1234)
        Random.seed!(seed)
        stopping_time = 0
        t = 1
        # make sure the constraint is sometimes binding
        @assert length(w) - 1 ∈ support(dist) && w̄ <= w[end]
        while true
            # Generate a wage draw
            w_val = w[rand(dist)] # the wage dist set up earlier
            if w_val ≥ w̄
                stopping_time = t
                break
            else
                t += 1
            end
        end
        return stopping_time
    end

    compute_mean_stopping_time(w̄, num_reps=10000) = mean(i ->
                                                             compute_stopping_time(w̄,
                                                             seed = i), 1:num_reps)
    c_vals = range(10,  40, length = 25)
    stop_times = similar(c_vals)

    beta = 0.99
    for (i, c) in enumerate(c_vals)
        w̄ = compute_reservation_wage_ψ(c, beta)
        stop_times[i] = compute_mean_stopping_time(w̄)
    end

    plot(c_vals, stop_times, label = "mean unemployment duration",
         xlabel = "unemployment compensation", ylabel = "months")

.. code-block:: julia
    :class: test

    # Just eyeball the plot pending undeprecation and rewrite.
    @testset begin
        @test stop_times[4] ≈ 8.1822
    end
