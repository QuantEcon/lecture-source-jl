.. _wald_friedman:

.. include:: /_static/includes/lecture_howto_jl.raw

.. highlight:: julia


***********************************************
:index:`A Problem that Stumped Milton Friedman`
***********************************************

(and that Abraham Wald solved by inventing sequential analysis)

.. index::
    single: Models; Sequential analysis

.. contents:: :depth: 2


Co-authored with Chase Coleman.

Overview
=========

This lecture describes a statistical decision problem encountered  by Milton Friedman and W. Allen Wallis during World War II when they were analysts at the U.S. Government's  Statistical Research Group at Columbia University

This problem led Abraham Wald :cite:`Wald47` to formulate **sequential analysis**, an approach to statistical decision problems intimately related to dynamic programming

In this lecture, we apply dynamic programming algorithms to Friedman and Wallis and Wald's problem

Key ideas in play will be:

-  Bayes' Law

-  Dynamic programming

-  Type I and type II statistical errors

   -  a type I error occurs when you reject a null hypothesis that is true

   -  a type II error is when you accept a null hypothesis that is false

-  Abraham Wald's **sequential probability ratio test**

-  The **power** of a statistical test

-  The **critical region** of a statistical test

-  A **uniformly most powerful test**





Origin of the problem
======================

On pages 137-139 of his 1998 book *Two Lucky People* with Rose Friedman :cite:`Friedman98`,
Milton Friedman described a problem presented to him and Allen Wallis
during World War II, when they worked at the US Government's
Statistical Research Group at Columbia University

Let's listen to Milton Friedman tell us what happened

"In order to understand the story, it is necessary to have an idea of a
simple statistical problem, and of the standard procedure for dealing
with it. The actual problem out of which sequential analysis grew will
serve. The Navy has two alternative designs (say A and B) for a
projectile. It wants to determine which is superior. To do so it
undertakes a series of paired firings. On each round it assigns the
value 1 or 0 to A accordingly as its performance is superior or inferio
to that of B and conversely 0 or 1 to B. The Navy asks the statistician
how to conduct the test and how to analyze the results.

"The standard statistical answer was to specify a number of firings (say
1,000) and a pair of percentages (e.g., 53% and 47%) and tell the client
that if A receives a 1 in more than 53% of the firings, it can be
regarded as superior; if it receives a 1 in fewer than 47%, B can be
regarded as superior; if the percentage is between 47% and 53%, neither
can be so regarded.

"When Allen Wallis was discussing such a problem with (Navy) Captain
Garret L. Schyler, the captain objected that such a test, to quote from
Allen's account, may prove wasteful. If a wise and seasoned ordnance
officer like Schyler were on the premises, he would see after the first
few thousand or even few hundred [rounds] that the experiment need not
be completed either because the new method is obviously inferior or
because it is obviously superior beyond what was hoped for
:math:`\ldots` "

Friedman and Wallis struggled with the problem but, after realizing that
they were not able to solve it,  described the problem to  Abraham Wald

That started Wald on the path that led him  to *Sequential Analysis* :cite:`Wald47`

We'll formulate the problem using dynamic programming



A dynamic programming approach
================================

The following presentation of the problem closely follows Dmitri
Berskekas's treatment in **Dynamic Programming and Stochastic Control** :cite:`Bertekas75`

A decision maker observes iid draws of a random variable :math:`z`

He (or she) wants to know which of two probability distributions :math:`f_0` or :math:`f_1` governs :math:`z`

After a number of draws, also to be determined, he makes a decision as to which of the distributions is generating the draws he observers

To help formalize the problem, let :math:`x \in \{x_0, x_1\}` be a hidden state that indexes the two distributions:

.. math::

    \mathbb P\{z = v \mid x \}
    = \begin{cases}
        f_0(v) & \mbox{if } x = x_0, \\
        f_1(v) & \mbox{if } x = x_1
    \end{cases}


Before observing any outcomes, the decision maker believes that the probability that :math:`x = x_0` is

.. math::

    p_{-1} =
    \mathbb P \{ x=x_0 \mid \textrm{ no observations} \} \in (0, 1)


After observing :math:`k+1` observations :math:`z_k, z_{k-1}, \ldots, z_0`, he updates this value to

.. math::

    p_k = \mathbb P \{ x = x_0 \mid z_k, z_{k-1}, \ldots, z_0 \},


which is calculated recursively by applying Bayes' law:

.. math::

    p_{k+1} = \frac{ p_k f_0(z_{k+1})}{ p_k f_0(z_{k+1}) + (1-p_k) f_1 (z_{k+1}) },
    \quad k = -1, 0, 1, \ldots


After observing :math:`z_k, z_{k-1}, \ldots, z_0`, the decision maker believes that :math:`z_{k+1}` has probability distribution

.. math::

    f(v) = p_k f_0(v) + (1-p_k) f_1 (v)


This is a mixture of distributions :math:`f_0` and :math:`f_1`, with the weight on :math:`f_0` being the posterior probability that :math:`x = x_0` [#f1]_

To help illustrate this kind of distribution, let's inspect some mixtures of beta distributions

The density of a beta probability distribution with parameters :math:`a` and :math:`b` is

.. math::

    f(z; a, b) = \frac{\Gamma(a+b) z^{a-1} (1-z)^{b-1}}{\Gamma(a) \Gamma(b)}
    \quad \text{where} \quad
    \Gamma(t) := \int_{0}^{\infty} x^{t-1} e^{-x} dx


We'll discretize this distribution to make it more straightforward to work with

The next figure shows two discretized beta distributions in the top panel

The bottom panel presents mixtures of these distributions, with various mixing probabilities :math:`p_k`

Activate the project environment, ensuring that ``Project.toml`` and ``Manifest.toml`` are in the same location as your notebook

.. code-block:: julia

    using Pkg; Pkg.activate(@__DIR__); #activate environment in the notebook's location


.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

  using Distributions, Plots, LaTeXStrings, LinearAlgebra, QuantEcon, Printf, Interpolations


  f0 = pdf.(Ref(Beta(1, 1)), range(0, stop = 1, length = 50))
  f0 = f0 / sum(f0)
  f1 = pdf.(Ref(Beta(9, 9)), range(0, stop = 1, length = 50))
  f1 = f1 / sum(f1)  # Make sure sums to 1

  a = plot([f0 f1],
      xlabel=L"$k$ Values",
      ylabel=L"Probability of $z_k$",
      labels=[L"$f_0$" L"$f_1$"],
      linewidth=2,
      ylims=[0.;0.07],
      title="Original Distributions")

  mix = zeros(50, 3)
  labels = Vector{String}(undef, 3)
  p_k = [0.25; 0.5; 0.75]
  for i in 1:3
      mix[:, i] = p_k[i] * f0 + (1 - p_k[i]) * f1
      labels[i] = string(L"$p_k$ = ", p_k[i])
  end

  b = plot(mix,
           xlabel = L"$k$ Values",
           ylabel = L"Probability of $z_k$",
           labels = labels,
           linewidth = 2,
           ylims = [0.;0.06],
           title = "Mixture of Original Distributions")

  plot(a, b, layout = (2, 1), size = (800, 600))

.. code-block:: julia
    :class: test

    @testset "First Plot Tests" begin
        @test mix[7] ≈ 0.005059526866095018
        @test all(isequal(0.02), f0)
        @test f1[10] ≈ 0.0011405494355950728
    end


Losses and costs
-------------------

After observing :math:`z_k, z_{k-1}, \ldots, z_0`, the decision maker
chooses among three distinct actions:


-  He decides that :math:`x = x_0` and draws no more :math:`z`'s

-  He decides that :math:`x = x_1` and draws no more :math:`z`'s

-  He postpones deciding now and instead chooses to draw a
   :math:`z_{k+1}`

Associated with these three actions, the decision maker can suffer three
kinds of losses:

-  A loss :math:`L_0` if he decides :math:`x = x_0` when actually
   :math:`x=x_1`

-  A loss :math:`L_1` if he decides :math:`x = x_1` when actually
   :math:`x=x_0`

-  A cost :math:`c` if he postpones deciding and chooses instead to draw
   another :math:`z`


Digression on type I and type II errors
----------------------------------------

If we regard  :math:`x=x_0` as a null hypothesis and :math:`x=x_1` as an alternative hypothesis,
then :math:`L_1` and :math:`L_0` are losses associated with two types of statistical errors.

- a type I error is an incorrect rejection of a true null hypothesis (a "false positive")

- a type II error is a failure to reject a false null hypothesis (a "false negative")

So when we treat :math:`x=x_0` as the null hypothesis

-  We can think of :math:`L_1` as the loss associated with a type I
   error

-  We can think of :math:`L_0` as the loss associated with a type II
   error


Intuition
-------------------

Let's try to guess what an optimal decision rule might look like before we go further

Suppose at some given point in time that :math:`p` is close to 1

Then our prior beliefs and the evidence so far point strongly to :math:`x = x_0`

If, on the other hand, :math:`p` is close to 0, then :math:`x = x_1` is strongly favored

Finally, if :math:`p` is in the middle of the interval :math:`[0, 1]`, then we have little information in either direction

This reasoning suggests a decision rule such as the one shown in the figure

.. figure:: /_static/figures/wald_dec_rule.png
    :scale: 40%


As we'll see, this is indeed the correct form of the decision rule

The key problem is to determine the threshold values :math:`\alpha, \beta`,
which will depend on the parameters listed above

You might like to pause at this point and try to predict the impact of a
parameter such as :math:`c` or :math:`L_0` on :math:`\alpha` or :math:`\beta`

A Bellman equation
-------------------

Let :math:`J(p)` be the total loss for a decision maker with current belief :math:`p` who chooses optimally

With some thought, you will agree that :math:`J` should satisfy the Bellman equation

.. math::
    :label: new1

    J(p) =
        \min
        \left\{
            (1-p) L_0, \; p L_1, \;
            c + \mathbb E [ J (p') ]
        \right\}


where :math:`p'` is the random variable defined by

.. math::

    p' = \frac{ p f_0(z)}{ p f_0(z) + (1-p) f_1 (z) }


when :math:`p` is fixed and :math:`z` is drawn from the current best guess, which is the distribution :math:`f` defined by

.. math::

    f(v) = p f_0(v) + (1-p) f_1 (v)


In the Bellman equation, minimization is over three actions:

#. accept :math:`x_0`
#. accept :math:`x_1`
#. postpone deciding and draw again

Let

.. math::

    A(p)
    := \mathbb E [ J (p') ]


Then we can represent the  Bellman equation as

.. math::

    J(p) =
    \min \left\{ (1-p) L_0, \; p L_1, \; c + A(p) \right\}


where :math:`p \in [0,1]`

Here

-  :math:`(1-p) L_0` is the expected loss associated with accepting
   :math:`x_0` (i.e., the cost of making a type II error)

-  :math:`p L_1` is the expected loss associated with accepting
   :math:`x_1` (i.e., the cost of making a type I error)

-  :math:`c + A(p)` is the expected cost associated with drawing one more :math:`z`



The optimal decision rule is characterized by two numbers :math:`\alpha, \beta \in (0,1) \times (0,1)` that satisfy

.. math::

    (1- p) L_0 < \min \{ p L_1, c + A(p) \}  \textrm { if } p \geq \alpha


and

.. math::

    p L_1 < \min \{ (1-p) L_0,  c + A(p) \} \textrm { if } p \leq \beta


The optimal decision rule is then

.. math::

    \textrm { accept } x=x_0 \textrm{ if } p \geq \alpha \\
    \textrm { accept } x=x_1 \textrm{ if } p \leq \beta \\
    \textrm { draw another }  z \textrm{ if }  \beta \leq p \leq \alpha


Our aim is to compute the value function :math:`J`, and from it the associated cutoffs :math:`\alpha`
and :math:`\beta`

One sensible approach is to write the three components of :math:`J`
that appear on the right side of the Bellman equation as separate functions

Later, doing this will help us obey **the don't repeat yourself (DRY)** golden rule of coding


Implementation
==================

Let's code this problem up and solve it

To approximate the value function that solves Bellman equation :eq:`new1`, we
use value function iteration

* For earlier examples of this technique see the :doc:`shortest path <short_path>`, :doc:`job search <mccall_model>` or :doc:`optimal growth <optgrowth>` lectures

As in the :doc:`optimal growth lecture <optgrowth>`, to approximate a continuous value function

* We iterate at a finite grid of possible values of :math:`p`

* When we evaluate :math:`A(p)` between grid points, we use linear interpolation

This means that to evaluate :math:`J(p)` where :math:`p` is not a grid point, we must use two points:

* First, we use the largest of all the grid points smaller than :math:`p`, and call it :math:`p_i`

* Second, we use the grid point immediately after :math:`p`, named :math:`p_{i+1}`, to approximate the function value as

.. math::

    J(p) = J(p_i) + (p - p_i) \frac{J(p_{i+1}) - J(p_i)}{p_{i+1} - p_{i}}


In one dimension, you can think of this as simply drawing a line between each pair of points on the grid

Here's the code

.. code-block:: julia

    expect_loss_choose_0(p, L0) = (1 - p) * L0

    expect_loss_choose_1(p, L1) = p * L1

    function EJ(p, f0, f1, J)
        # Get the current distribution we believe (p * f0 + (1 - p) * f1)
        curr_dist = p * f0 + (1 - p) * f1

        # Get tomorrow's expected distribution through Bayes law
        tp1_dist = clamp.((p * f0) ./ (p * f0 + (1 - p) * f1), 0, 1)

        # Evaluate the expectation
        EJ = dot(curr_dist, J.(tp1_dist))

        return EJ
    end

    expect_loss_cont(p, c, f0, f1, J) = c + EJ(p, f0, f1, J)

    function bellman_operator(pgrid, c, f0, f1, L0, L1, J)
        m = length(pgrid)
        @assert m == length(J)

        J_out = zeros(m)
        J_interp = LinearInterpolation(pgrid, J) # The method from Interpolations.jl

        for (p_ind, p) in enumerate(pgrid)
            # Payoff of choosing model 0
            p_c_0 = expect_loss_choose_0(p, L0)
            p_c_1 = expect_loss_choose_1(p, L1)
            p_con = expect_loss_cont(p, c, f0, f1, J_interp)

            J_out[p_ind] = min(p_c_0, p_c_1, p_con)
        end

        return J_out
    end

    # Create two distributions over 50 values for k
    # We are using a discretized beta distribution

    p_m1 = range(0, stop = 1, length = 50)
    f0 = clamp.(pdf.(Ref(Beta(1, 1)), p_m1), 1e-8, Inf)
    f0 = f0 / sum(f0)
    f1 = clamp.(pdf.(Ref(Beta(9, 9)), p_m1), 1e-8, Inf)
    f1 = f1 / sum(f1)

    # To solve
    pg = range(0, stop = 1, length = 251)
    J1 = compute_fixed_point(x -> bellman_operator(pg, 0.5, f0, f1, 5.0, 5.0, x),
        zeros(length(pg)), err_tol = 1e-6, print_skip = 5);

.. code-block:: julia
    :class: test

    @testset "Second Block Tests" begin
        @test J1[19] == 0.36
        @test f1[40] ≈ 0.002163769345396983
        @test length(pg) == 251 && pg[1] == 0 && pg[end] == 1
    end

Running it produces the following output on our machine


The distance column shows the maximal distance between successive iterates

This converges to zero quickly, indicating a successful iterative procedure

Iteration terminates when the distance falls below some threshold


A more sophisticated implementation
-------------------------------------

Now for some gentle criticisms of the preceding code

By writing the code in terms of functions, we have to pass around
some things that are constant throughout the problem

* :math:`c`, :math:`f_0`, :math:`f_1`, :math:`L_0`, and :math:`L_1`

So now let's turn our simple script into a type

This will allow us to simplify the function calls and make the code more reusable


We shall construct two types that

* store all of our parameters for us internally

* represent the solution to our Bellman equation alongside the :math:`\alpha` and :math:`\beta` decision cutoffs

* accompany many of the same functions used above which now act on the type directly

* allow us, in addition, to simulate draws and the decision process under different prior beliefs


.. code-block:: julia

    mutable struct WFSolution{TAV <: AbstractVector, TR<:Real}
        J::TAV
        lb::TR
        ub::TR
    end

    struct WaldFriedman{TR <: Real,
                        TI <: Integer,
                        TAV1 <: AbstractVector,
                        TAV2 <: AbstractVector}
        c::TR
        L0::TR
        L1::TR
        f0::TAV1
        f1::TAV1
        m::TI
        pgrid::TAV2
        sol::WFSolution
    end

    function WaldFriedman(c, L0, L1, f0, f1; m = 25)

        pgrid = range(0.0, stop = 1.0, length = m)

        # Renormalize distributions so nothing is "too" small
        f0 = clamp.(f0, 1e-8, 1-1e-8)
        f1 = clamp.(f1, 1e-8, 1-1e-8)
        f0 = f0 / sum(f0)
        f1 = f1 / sum(f1)
        J = zeros(m)
        lb = 0.
        ub = 0.

        WaldFriedman(c, L0, L1, f0, f1, m, pgrid, WFSolution(J, lb, ub))
    end

    current_distribution(wf::WaldFriedman, p::Real) = p * wf.f0 + (1 - p) * wf.f1

    function bayes_update_k(wf, p, k)
        f0_k = wf.f0[k]
        f1_k = wf.f1[k]

        p_tp1 = p * f0_k / (p * f0_k + (1 - p) * f1_k)

        return clamp(p_tp1, 0, 1)
    end

    bayes_update_all(wf, p) = clamp.(p * wf.f0 ./ (p * wf.f0 + (1 - p) * wf.f1), 0, 1)

    payoff_choose_f0(wf, p) = (1 - p) * wf.L0

    payoff_choose_f1(wf, p) = p * wf.L1

    function EJ(wf, p, J)
        # Pull out information
        f0, f1 = wf.f0, wf.f1

        # Get the current believed distribution and tomorrows possible dists
        # Need to clip to make sure things don't blow up (go to infinity)
        curr_dist = current_distribution(wf, p)
        tp1_dist = bayes_update_all(wf, p)

        # Evaluate the expectation
        EJ = dot(curr_dist, J.(tp1_dist))

        return EJ
    end

    payoff_continue(wf, p, J) = wf.c + EJ(wf, p, J)

    function bellman_operator(wf, J)
        c, L0, L1, f0, f1 = wf.c, wf.L0, wf.L1, wf.f0, wf.f1
        m, pgrid = wf.m, wf.pgrid

        J_out = similar(J)
        J_interp = LinearInterpolation(pgrid, J)

        for (p_ind, p) in enumerate(pgrid)
            # Payoff of choosing model 0
            p_c_0 = payoff_choose_f0(wf, p)
            p_c_1 = payoff_choose_f1(wf, p)
            p_con = payoff_continue(wf, p, J_interp)

            J_out[p_ind] = min(p_c_0, p_c_1, p_con)
        end

        return J_out
    end

    function find_cutoff_rule(wf, J)
        m, pgrid = wf.m, wf.pgrid

        # Evaluate cost at all points on grid for choosing a model
        p_c_0 = payoff_choose_f0.(Ref(wf), pgrid)
        p_c_1 = payoff_choose_f1.(Ref(wf), pgrid)

        # The cutoff points can be found by differencing these costs with
        # the Bellman equation (J is always less than or equal to p_c_i)
        lb = pgrid[searchsortedlast(p_c_1 - J, 1e-10)]
        ub = pgrid[searchsortedlast(J - p_c_0, -1e-10)]

        return lb, ub
    end

    function solve_model!(wf; tol = 1e-7)
        bell_op(x) = bellman_operator(wf, x)
        J =  compute_fixed_point(bell_op, zeros(wf.m), err_tol=tol, print_skip=5)

        wf.sol.J = J
        wf.sol.lb, wf.sol.ub = find_cutoff_rule(wf, J)
        return J
    end

    function simulate(wf, f; p0 = 0.5)
        # Check whether vf is computed
        if sum(abs, wf.sol.J) < 1e-8
            solve_model!(wf)
        end

        # Unpack useful info
        lb, ub = wf.sol.lb, wf.sol.ub
        drv = DiscreteRV(f)

        # Initialize a couple useful variables
        decision = 0
        p = p0
        t = 0

        while true
            # Maybe should specify which distribution is correct one so that
            # the draws come from the "right" distribution
            k = rand(drv)
            t = t + 1
            p = bayes_update_k(wf, p, k)
            if p < lb
                decision = 1
                break
            elseif p > ub
                decision = 0
                break
            end
        end

        return decision, p, t
    end

    abstract type HiddenDistribution end
    struct F0 <: HiddenDistribution end
    struct F1 <: HiddenDistribution end

    function simulate_tdgp(wf, f; p0 = 0.5)
        decision, p, t = simulate(wf, wf.f0; p0=p0)

        correct = (decision == 0)

        return correct, p, t
    end

    function simulate_tdgp(wf, f; p0 = 0.5)
        decision, p, t = simulate(wf, wf.f1; p0=p0)

        correct = (decision == 1)

        return correct, p, t
    end

    function stopping_dist(wf; ndraws = 250, f = F0())
        # Allocate space
        tdist = fill(0, ndraws)
        cdist = fill(false, ndraws)

        for i in 1:ndraws
            correct, p, t = simulate_tdgp(wf, f)
            tdist[i] = t
            cdist[i] = correct
        end

        return cdist, tdist
    end


Now let's use our type to solve Bellman equation :eq:`new1` and verify that it gives similar output


.. code-block:: julia

    # Create two distributions over 50 values for k
    # We are using a discretized beta distribution

    p_m1 = range(0, stop = 1, length = 50)
    f0 = clamp.(pdf.(Ref(Beta(1, 1)), p_m1), 1e-8, Inf)
    f0 = f0 / sum(f0)
    f1 = clamp.(pdf.(Ref(Beta(9, 9)), p_m1), 1e-8, Inf)
    f1 = f1 / sum(f1);

    wf = WaldFriedman(0.5, 5.0, 5.0, f0, f1; m=251)
    J2 = compute_fixed_point(x -> bellman_operator(wf, x), zeros(wf.m), err_tol=1e-6, print_skip=5)

    @printf("If this is true then both approaches gave same answer:\n")
    print(isapprox(J1, J2; atol=1e-5))

.. code-block:: julia
    :class: test

    @testset "Second Calculation Tests" begin
        @test J1 ≈ J2 atol = 1e-5
    end

We get the same output in terms of distance


The approximate value functions produced are also the same

Rather than discuss this further, let's go ahead and use our code to generate some results


Analysis
=====================

Now that our routines are working, let's inspect the solutions

We'll start with the following parameterization


.. code-block:: julia

  function analysis_plot(;c = 1.25, L0 = 27.0, L1 = 27.0, a0 = 2.5, b0 = 2.0,
                         a1 = 2.0, b1 = 2.5, m = 251)
      f0 = pdf.(Ref(Beta(a0, b0)), range(0, stop = 1, length = m))
      f0 = f0 / sum(f0)
      f1 = pdf.(Ref(Beta(a1, b1)), range(0, stop = 1, length = m))
      f1 = f1 / sum(f1)  # Make sure sums to 1

      # Create an instance of our WaldFriedman class
      wf = WaldFriedman(c, L0, L1, f0, f1; m = m);

      # Solve and simulate the solution
      cdist, tdist = stopping_dist(wf; ndraws = 5000)

      a = plot([f0 f1],
          xlabel = L"$k$ Values",
          ylabel = L"Probability of $z_k$",
          labels = [L"$f_0$" L"$f_1$"],
          linewidth = 2,
          title= " Distributions over Outcomes")

      b = plot(wf.pgrid, wf.sol.J,
               xlabel = L"$p_k$",
               ylabel = "Value of Bellman",
               linewidth = 2,
               title = "Bellman Equation")
      plot!(fill(wf.sol.lb, 2), [minimum(wf.sol.J); maximum(wf.sol.J)],
            linewidth = 2, color = :black, linestyle = :dash, label = "",
            ann = (wf.sol.lb-0.05, 5., L"\beta"))
      plot!(fill(wf.sol.ub, 2), [minimum(wf.sol.J); maximum(wf.sol.J)],
            linewidth = 2, color = :black, linestyle = :dash, label = "",
            ann = (wf.sol.ub+0.02, 5., L"\alpha"), legend = :none)

      counts = [sum(tdist .== i) for i in 1:maximum(tdist)]

      c = bar(counts,
              xticks = 0:1:maximum(tdist),
              xlabel = "Time",
              ylabel = "Frequency",
              title = "Stopping Times",
              legend = :none)

      counts = [sum(cdist .== i-1) for i in 1:2]

      d = bar([0; 1],
              counts,
              xticks = [0; 1],
              title = "Correct Decisions",
              ann = (0.0, 0.6 * sum(cdist),
                     "Percent Correct = $(sum(cdist)/length(cdist))"),
              legend = :none)

      plot(a, b, c, d, layout = (2, 2), size = (1200, 800))
  end

  analysis_plot()

.. code-block:: julia
    :class: test

    # These tests need to be eyeballed, AFAIK, since the function above doesn't return anything.


The code to generate this figure can be found in `wald_solution_plots.jl <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/wald_friedman/wald_solution_plots.jl>`__

Value Function
-----------------

In the top left subfigure we have the two beta distributions, :math:`f_0` and :math:`f_1`

In the top right we have corresponding value function :math:`J`

It equals :math:`p L_1` for :math:`p \leq \beta`, and :math:`(1-p )L_0` for :math:`p
\geq \alpha`

The slopes of the two linear pieces of the value function are determined by :math:`L_1`
and :math:`- L_0`

The value function is smooth in the interior region, where the posterior
probability assigned to :math:`f_0` is in the indecisive region :math:`p \in (\beta, \alpha)`

The decision maker continues to sample until the probability that he attaches to model :math:`f_0` falls below :math:`\beta` or above :math:`\alpha`


Simulations
-----------------

The bottom two subfigures show the outcomes of 500 simulations of the decision process

On the left is a histogram of the stopping times, which equal the number of draws of :math:`z_k` required to make a decision

The average number of draws is around 6.6

On the right is the fraction of correct decisions at the stopping time

In this case the decision maker is correct 80% of the time


Comparative statics
----------------------

Now let's consider the following exercise

We double the cost of drawing an additional observation

Before you look, think about what will happen:

-  Will the decision maker be correct more or less often?

-  Will he make decisions sooner or later?


.. code-block:: julia

  analysis_plot(c = 2.5)


Notice what happens

The stopping times dropped dramatically!

Increased cost per draw has induced the decision maker usually to take only 1 or 2 draws before deciding

Because he decides with less, the percentage of time he is correct drops

This leads to him having a higher expected loss when he puts equal weight on both models


A notebook implementation
---------------------------


To facilitate comparative statics, we provide a `Jupyter notebook <http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/Wald_Friedman_jl.ipynb>`__ that generates the same plots, but with sliders


With these sliders you can adjust parameters and immediately observe

*  effects on the smoothness of the value function in the indecisive middle range as we increase the number of grid points in the piecewise linear  approximation.

* effects of different settings for the cost parameters :math:`L_0, L_1, c`, the parameters of two beta distributions :math:`f_0` and :math:`f_1`, and the number of points and linear functions :math:`m` to use in the piece-wise continuous approximation to the value function.

* various simulations from :math:`f_0` and associated distributions of waiting times to making a decision

* associated histograms of correct and incorrect decisions


Comparison with Neyman-Pearson formulation
=============================================

For several reasons, it is useful to describe the theory underlying the test
that Navy Captain G. S. Schuyler had been told to use and that led him
to approach Milton Friedman and Allan Wallis to convey his conjecture
that superior practical procedures existed

Evidently, the Navy had told
Captail Schuyler to use what it knew to be a state-of-the-art
Neyman-Pearson test

We'll rely on Abraham Wald's :cite:`Wald47` elegant summary of Neyman-Pearson theory

For our purposes, watch for there features of the setup:

-  the assumption of a *fixed* sample size :math:`n`

-  the application of laws of large numbers, conditioned on alternative
   probability models, to interpret the probabilities :math:`\alpha` and
   :math:`\beta` defined in the Neyman-Pearson theory

Recall that in the sequential analytic formulation above, that

-  The sample size :math:`n` is not fixed but rather an object to be
   chosen; technically :math:`n` is a random variable

-  The parameters :math:`\beta` and :math:`\alpha` characterize cut-off
   rules used to determine :math:`n` as a random variable

-  Laws of large numbers make no appearances in the sequential
   construction

In chapter 1 of **Sequential Analysis** :cite:`Wald47` Abraham Wald summarizes the
Neyman-Pearson approach to hypothesis testing

Wald frames the problem as making a decision about a probability
distribution that is partially known

(You have to assume that *something* is already known in order to state a well posed problem.
Usually, *something* means *a lot*.)

By limiting  what is unknown, Wald uses the following simple structure
to illustrate the main ideas.

-  a decision maker wants to decide which of two distributions
   :math:`f_0`, :math:`f_1` govern an i.i.d. random variable :math:`z`

-  The null hypothesis :math:`H_0` is the statement that :math:`f_0`
   governs the data.

-  The alternative hypothesis :math:`H_1` is the statement that
   :math:`f_1` governs the data.

-  The problem is to devise and analyze a test of hypothesis
   :math:`H_0` against the alternative hypothesis :math:`H_1` on the
   basis of a sample of a fixed number :math:`n` independent
   observations :math:`z_1, z_2, \ldots, z_n` of the random variable
   :math:`z`.

To quote Abraham Wald,

-  A test procedure leading to the acceptance or rejection of the
   hypothesis in question is simply a rule specifying, for each possible
   sample of size :math:`n`, whether the hypothesis should be accepted
   or rejected on the basis of the sample. This may also be expressed as
   follows: A test procedure is simply a subdivision of the totality of
   all possible samples of size :math:`n` into two mutually exclusive
   parts, say part 1 and part 2, together with the application of the
   rule that the hypothesis be accepted if the observed sample is
   contained in part 2. Part 1 is also called the critical region. Since
   part 2 is the totality of all samples of size 2 which are not
   included in part 1, part 2 is uniquely determined by part 1. Thus,
   choosing a test procedure is equivalent to determining a critical
   region.

Let's listen to Wald longer:

-  As a basis for choosing among critical regions the following
   considerations have been advanced by Neyman and Pearson: In accepting
   or rejecting :math:`H_0` we may commit errors of two kinds. We commit
   an error of the first kind if we reject :math:`H_0` when it is true;
   we commit an error of the second kind if we accept :math:`H_0` when
   :math:`H_1` is true. After a particular critical region :math:`W` has
   been chosen, the probability of committing an error of the first
   kind, as well as the probability of committing an error of the second
   kind is uniquely determined. The probability of committing an error
   of the first kind is equal to the probability, determined by the
   assumption that :math:`H_0` is true, that the observed sample will be
   included in the critical region :math:`W`. The probability of
   committing an error of the second kind is equal to the probability,
   determined on the assumption that :math:`H_1` is true, that the
   probability will fall outside the critical region :math:`W`. For any
   given critical region :math:`W` we shall denote the probability of an
   error of the first kind by :math:`\alpha` and the probability of an
   error of the second kind by :math:`\beta`.

Let's listen carefully to how Wald applies a law of large numbers to
interpret :math:`\alpha` and :math:`\beta`:

-  The probabilities :math:`\alpha` and :math:`\beta` have the
   following important practical interpretation: Suppose that we draw a
   large number of samples of size :math:`n`. Let :math:`M` be the
   number of such samples drawn. Suppose that for each of these
   :math:`M` samples we reject :math:`H_0` if the sample is included in
   :math:`W` and accept :math:`H_0` if the sample lies outside
   :math:`W`. In this way we make :math:`M` statements of rejection or
   acceptance. Some of these statements will in general be wrong. If
   :math:`H_0` is true and if :math:`M` is large, the probability is
   nearly :math:`1` (i.e., it is practically certain) that the
   proportion of wrong statements (i.e., the number of wrong statements
   divided by :math:`M`) will be approximately :math:`\alpha`. If
   :math:`H_1` is true, the probability is nearly :math:`1` that the
   proportion of wrong statements will be approximately :math:`\beta`.
   Thus, we can say that in the long run [ here Wald applies a law of
   large numbers by driving :math:`M \rightarrow \infty` (our comment,
   not Wald's) ] the proportion of wrong statements will be
   :math:`\alpha` if :math:`H_0`\ is true and :math:`\beta` if
   :math:`H_1` is true.

The quantity :math:`\alpha` is called the *size* of the critical region,
and the quantity :math:`1-\beta` is called the *power* of the critical
region.

Wald notes that

-  one critical region :math:`W` is more desirable than another if it
   has smaller values of :math:`\alpha` and :math:`\beta`. Although
   either :math:`\alpha` or :math:`\beta` can be made arbitrarily small
   by a proper choice of the critical region :math:`W`, it is possible
   to make both :math:`\alpha` and :math:`\beta` arbitrarily small for a
   fixed value of :math:`n`, i.e., a fixed sample size.

Wald summarizes Neyman and Pearson's setup as follows:

-  Neyman and Pearson show that a region consisting of all samples
   :math:`(z_1, z_2, \ldots, z_n)` which satisfy the inequality

    .. math::

        \frac{ f_1(z_1) \cdots f_1(z_n)}{f_0(z_1) \cdots f_1(z_n)} \geq k


   is a most powerful critical region for testing the hypothesis
   :math:`H_0` against the alternative hypothesis :math:`H_1`. The term
   :math:`k` on the right side is a constant chosen so that the region
   will have the required size :math:`\alpha`.


Wald goes on to discuss Neyman and Pearson's concept of *uniformly most
powerful* test.

Here is how Wald introduces the notion of a sequential test

-  A rule is given for making one of the following three decisions at any stage of
   the experiment (at the m th trial for each integral value of m ): (1) to
   accept the hypothesis H , (2) to reject the hypothesis H , (3) to
   continue the experiment by making an additional observation. Thus, such
   a test procedure is carried out sequentially. On the basis of the first
   observation one of the aforementioned decisions is made. If the first or
   second decision is made, the process is terminated. If the third
   decision is made, a second trial is performed. Again, on the basis of
   the first two observations one of the three decisions is made. If the
   third decision is made, a third trial is performed, and so on. The
   process is continued until either the first or the second decisions is
   made. The number n of observations required by such a test procedure is
   a random variable, since the value of n depends on the outcome of the
   observations.

.. rubric:: Footnotes

.. [#f1] Because the decision maker believes that :math:`z_{k+1}` is
    drawn from a mixture of two i.i.d. distributions, he does *not*
    believe that the sequence :math:`[z_{k+1}, z_{k+2}, \ldots]` is i.i.d.
    Instead, he believes that it is *exchangeable*. See :cite:`Kreps88`
    chapter 11, for a discussion of exchangeability.
