.. _smoothing:

.. include:: /_static/includes/lecture_howto_jl_full.raw

.. highlight:: julia

*******************************************************************
Consumption and Tax Smoothing with Complete and Incomplete Markets
*******************************************************************

.. index::
    single: Consumption; Tax

.. contents:: :depth: 2

Overview
========

This lecture describes two types of consumption-smoothing  and tax-smoothing models

* one is in the **complete markets** tradition of Lucas and Stokey :cite:`LucasStokey1983`

* the other is in the **incomplete markets** tradition  of Hall :cite:`Hall1978` and Barro :cite:`Barro1979`

*Complete markets* allow a consumer or government to buy or sell claims contingent on all possible states of the world

*Incomplete markets* allow a consumer or government to buy or sell only a limited set of securities, often only a single risk-free security

Hall :cite:`Hall1978` and Barro :cite:`Barro1979` both assumed that the only asset that can be traded is a risk-free one period bond

Hall assumed an exogenous stochastic process of nonfinancial income and
an exogenous gross interest rate on one period risk-free debt that equals
:math:`\beta^{-1}`, where :math:`\beta \in (0,1)` is also a consumer's
intertemporal discount factor

Barro :cite:`Barro1979` made an analogous assumption about the risk-free interest
rate in a tax-smoothing model that we regard as isomorphic to Hall's
consumption-smoothing model

We maintain Hall and Barro's assumption about the interest rate when we describe an
incomplete markets version of our model

In addition, we extend their assumption about the interest rate to an appropriate counterpart that we use in a "complete markets" model in the style of
Lucas and Stokey :cite:`LucasStokey1983`

While we are equally interested in consumption-smoothing and tax-smoothing
models, for the most part we focus explicitly on consumption-smoothing
versions of these models

But for each version of the consumption-smoothing model there is a natural tax-smoothing counterpart obtained simply by

*  relabeling consumption as tax collections and nonfinancial income as government expenditures

*  relabeling the consumer's debt as the government's *assets*

For elaborations on this theme, please see :doc:`perm_income_cons` and later parts of this lecture

We'll consider two closely related alternative assumptions about the consumer's
exogenous nonfinancial income process (or in the tax-smoothing
interpretation, the government's exogenous expenditure process):

*  that it obeys a finite :math:`N` state Markov chain (setting :math:`N=2` most of the time)

*  that it is described by a linear state space model with a continuous
   state vector in :math:`{\mathbb R}^n` driven by a Gaussian vector iid shock
   process

We'll spend most of this lecture studying the finite-state Markov specification, but will briefly treat the linear state space specification before concluding

Relationship to Other Lectures
------------------------------

This lecture can be viewed as a followup to :doc:`perm_income_cons` and  a warm up for a model of tax smoothing described in :doc:`opt_tax_recur`

Linear-quadratic versions of the Lucas-Stokey tax-smoothing model are described in :doc:`lqramsey`

The key differences between those lectures and this one are

* Here the decision maker takes all prices as exogenous, meaning that his decisions do not affect them

* In :doc:`lqramsey` and :doc:`opt_tax_recur`, the decision maker -- the government in the case of these lectures -- recognizes that his decisions affect prices

So these later lectures are partly about how the government should  manipulate prices of government debt

Background
==========

Outcomes in consumption-smoothing (or tax-smoothing) models emerge from two
sources:

*  a decision maker -- a consumer in the consumption-smoothing model or
   a government in the tax-smoothing model -- who wants to maximize an
   intertemporal objective function that expresses its preference for
   paths of consumption (or tax collections) that are *smooth* in the
   sense of not varying across time and Markov states

*  a set of trading opportunities that allow the optimizer to transform
   a possibly erratic nonfinancial income (or government expenditure)
   process into a smoother consumption (or tax collections) process by
   purchasing or selling financial securities

In the complete markets version of the model, each period the consumer
can buy or sell one-period ahead state-contingent securities whose
payoffs depend on next period's realization of the Markov state

In the two-state Markov chain case, there are two such securities each period

In an :math:`N` state Markov state version of the model,  :math:`N` such securities are traded each period

These state-contingent securities are commonly called Arrow securities, after `Kenneth Arrow <https://en.wikipedia.org/wiki/Kenneth_Arrow>`__ who first theorized about them

In the incomplete markets version of the model, the consumer can buy and sell only one security each period, a risk-free bond with gross return :math:`\beta^{-1}`

Finite State Markov Income Process
----------------------------------

In each version of the consumption-smoothing model, nonfinancial income is governed by a two-state Markov chain (it's easy to generalize this to an :math:`N` state Markov chain)

In particular, the *state of the world* is given by :math:`s_t` that follows
a Markov chain with transition probability matrix

.. math::

    P_{ij} = \mathbb P \{s_{t+1} = \bar s_j \,|\, s_t = \bar s_i \}

Nonfinancial income :math:`\{y_t\}` obeys

.. math::

    y_t =
    \begin{cases}
        \bar y_1 & \quad \text{if } s_t = \bar s_1 \\
        \bar y_2 & \quad \text{if } s_t = \bar s_2
    \end{cases}

A consumer wishes to maximize

.. math::
    :label: cs_1

    \mathbb E
    \left[
        \sum_{t=0}^\infty \beta^t u(c_t)
    \right]
    \quad
    \text{where} \quad
    u(c_t) = - (c_t -\gamma)^2
    \quad \text{and} \quad
     0 < \beta < 1

Remark About Isomorphism
^^^^^^^^^^^^^^^^^^^^^^^^

We can regard these as Barro :cite:`Barro1979`  tax-smoothing models if we set
:math:`c_t = T_t` and :math:`G_t = y_t`, where :math:`T_t` is total tax
collections and :math:`\{G_t\}` is an exogenous government expenditures
process

Market Structure
----------------

The two models differ in how effectively the market structure allows the
consumer to transfer resources across time and Markov states, there
being more transfer opportunities in the complete markets setting than
in the incomplete markets setting

Watch how these differences in opportunities affect

-  how smooth consumption is across time and Markov states

-  how the consumer chooses to make his levels of indebtedness behave
   over time and across Markov states

Model 1 (Complete Markets)
==========================

At each date :math:`t \geq 0`, the consumer trades **one-period ahead
Arrow securities**

We assume that prices of these securities are exogenous to the consumer
(or in the tax-smoothing version of the model, to the government)

*Exogenous* means that they are unaffected by the  decision maker

In Markov state :math:`s_t` at time :math:`t`, one unit of consumption
in state :math:`s_{t+1}` at time :math:`t+1` costs :math:`q(s_{t+1} \,|\, s_t)` units of the time :math:`t` consumption good

At time :math:`t=0`, the consumer starts with an inherited level of debt
due at time :math:`0` of :math:`b_0` units of time :math:`0` consumption
goods

The consumer's budget constraint at :math:`t \geq 0` in Markov
state :math:`s_t` is

.. math::

    c_t + b_t
    \leq y(s_t) +
    \sum_j  q(\bar s_j \,|\, s_t ) \, b_{t+1}(\bar s_j \,|\, s_t)

where :math:`b_t` is the consumer's one-period debt that falls due at time :math:`t` and  :math:`b_{t+1}(\bar s_j\,|\, s_t)` are the consumer's time
:math:`t` sales of the  time :math:`t+1` consumption good in Markov state :math:`\bar s_j`, a source of time :math:`t` revenues

An analogue of Hall's assumption that the one-period risk-free gross
interest rate is :math:`\beta^{-1}` is

.. math::
    :label: cs_2

    q(\bar s_j \,|\, \bar s_i) = \beta P_{ij}

To understand this, observe that in state :math:`\bar s_i` it costs :math:`\sum_j q(\bar s_j \,|\, \bar s_i)`  to purchase one unit of consumption next period *for sure*, i.e., meaning no matter what state of the world  occurs at :math:`t+1`

Hence the implied price of a risk-free claim on one unit of consumption next
period is

.. math::

    \sum_j q(\bar s_j \,|\, \bar s_i) =  \sum_j \beta P_{ij} =  \beta

This confirms that :eq:`cs_2` is a natural analogue of Hall's assumption about the
risk-free one-period interest rate

First-order necessary conditions for maximizing the consumer's expected utility are

.. math::

    \beta \frac{u'(c_{t+1})}{u'(c_t) } \mathbb P\{s_{t+1}\,|\, s_t \}
        = q(s_{t+1} \,|\, s_t)

or, under our assumption :eq:`cs_2` on Arrow security prices,

.. math::
    :label: cs_3

    c_{t+1} = c_t

Thus, our consumer sets :math:`c_t = \bar c` for all :math:`t \geq 0` for some value :math:`\bar c` that it is our job now to determine

**Guess:** We'll make the plausible guess that

.. math::
    :label: eq_guess

    b_{t+1}(\bar s_j \,|\, s_t = \bar s_i) = b(\bar s_j) ,
            \quad i=1,2; \;\; j= 1,2

so that the amount borrowed today turns out to depend only on *tomorrow's* Markov state. (Why is this is a plausible guess?)

To determine :math:`\bar c`, we shall pursue the implications of the consumer's budget constraints in each Markov state today and  our guess :eq:`eq_guess` about the consumer's debt level choices

For :math:`t \geq 1`, these imply

.. math::
    :label: cs_4a

    \begin{aligned}
        \bar c + b(\bar s_1) & = y(\bar s_1) + q(\bar s_1\,|\, \bar s_1) b(\bar s_1) + q(\bar s_2 \,|\, \bar s_1)  b(\bar s_2) \cr
        \bar c + b(\bar s_2) & = y(\bar s_2) + q(\bar s_1\,|\, \bar s_2) b(\bar s_1) + q(\bar s_2 \,|\, \bar s_2) b(\bar s_2),
    \end{aligned}

or

.. math::

    \begin{bmatrix}
       b(\bar s_1) \cr b(\bar s_2)
    \end{bmatrix} +
    \begin{bmatrix}
    \bar c \cr \bar c
    \end{bmatrix}
    =
    \begin{bmatrix}
        y(\bar s_1) \cr y(\bar s_2)
    \end{bmatrix}
    + \beta
    \begin{bmatrix}
        P_{11} & P_{12} \cr P_{21} & P_{22}
    \end{bmatrix}
    \begin{bmatrix}
        b(\bar s_1) \cr b(\bar s_2)
    \end{bmatrix}

These are :math:`2` equations in the :math:`3` unknowns
:math:`\bar c, b(\bar s_1), b(\bar s_2)`

To get a third equation, we assume that at time :math:`t=0`, :math:`b_0`
is the debt due; and we assume that at time :math:`t=0`, the Markov
state is :math:`\bar s_1`

Then the budget constraint at time :math:`t=0` is

.. math::
    :label: cs_5

    \bar c + b_0 = y(\bar s_1) + q(\bar s_1 \,|\, \bar s_1) b(\bar s_1) + q(\bar s_2\,|\,\bar s_1) b(\bar s_2)

If we substitute  :eq:`cs_5` into the first equation of :eq:`cs_4a` and rearrange, we
discover that

.. math::
    :label: cs_6

    b(\bar s_1) = b_0

We can then use the second equation of :eq:`cs_4a`  to deduce the restriction

.. math::
    :label: cs_7

    y(\bar s_1) - y(\bar s_2) + [q(\bar s_1\,|\, \bar s_1) - q(\bar s_1\,|\, \bar s_2) - 1 ] b_0
      + [q(\bar s_2\,|\,\bar s_1) + 1 - q(\bar s_2 \,|\, \bar s_2) ] b(\bar s_2) = 0 ,

an equation in the unknown :math:`b(\bar s_2)`

Knowing :math:`b(\bar s_1)` and :math:`b(\bar s_2)`, we can solve equation :eq:`cs_5`  for the constant level of consumption :math:`\bar c`

Key outcomes
------------

The preceding calculations indicate that in the complete markets version
of our model, we obtain the following striking results:

*  The consumer chooses to make consumption perfectly constant across
   time and Markov states

We computed the constant level of consumption :math:`\bar c` and indicated how that level depends on the underlying specifications of preferences, Arrow securities prices,  the stochastic process of exogenous nonfinancial income, and the initial debt level :math:`b_0`

*  The consumer's debt neither accumulates, nor decumulates, nor drifts.
   Instead the debt level each period is an exact function of the Markov
   state, so in the two-state Markov case, it switches between two
   values

*  We have verified guess :eq:`eq_guess`

We computed how one of those debt levels depends entirely on initial debt -- it equals it -- and how the other value depends on virtually all  remaining parameters of the model

Code
-----

Here's some code that, among other things, contains a function called `consumption_complete()`

This function computes :math:`b(\bar s_1), b(\bar s_2), \bar c` as outcomes given a set of parameters, under the assumption of complete markets

Setup
------------------

.. literalinclude:: /_static/includes/deps.jl

.. code-block:: julia
  :class: test

  using Test

.. code-block:: julia

    using Parameters, Plots, QuantEcon, Random
    gr(fmt = :png)

.. code-block:: julia

    ConsumptionProblem = @with_kw (β = 0.96,
                                   y = [2.0, 1.5],
                                   b0 = 3.0,
                                   P = [0.8 0.2
                                        0.4 0.6])

    function consumption_complete(cp)

        @unpack β, P, y, b0 = cp   # Unpack

        y1, y2 = y                              # extract income levels
        b1 = b0                                 # b1 is known to be equal to b0
        Q = β * P                               # assumed price system

        # Using equation (7) calculate b2
        b2 = (y2 - y1 - (Q[1, 1] - Q[2, 1] - 1) * b1) / (Q[1, 2] + 1 - Q[2, 2])

        # Using equation (5) calculae c_bar
        c_bar = y1 - b0 + ([b1 b2] * Q[1, :])[1]

        return c_bar, b1, b2
    end

    function consumption_incomplete(cp; N_simul = 150)

        @unpack β, P, y, b0 = cp  # unpack

        # for the simulation use the MarkovChain type
        mc = MarkovChain(P)

        # useful variables
        y = y
        v = inv(I - β * P) * y

        # simulate state path
        s_path = simulate(mc, N_simul, init=1)

        # store consumption and debt path
        b_path, c_path = ones(N_simul + 1), ones(N_simul)
        b_path[1] = b0

        # optimal decisions from (12) and (13)
        db = ((1 - β) * v - y) / β

        for (i, s) in enumerate(s_path)
            c_path[i] = (1 - β) * (v[s, 1] - b_path[i])
            b_path[i + 1] = b_path[i] + db[s, 1]
        end

        return c_path, b_path[1:end - 1], y[s_path], s_path
    end

Let's test by checking that :math:`\bar c` and :math:`b_2` satisfy the budget constraint

.. code-block:: julia

    cp = ConsumptionProblem()
    c_bar, b1, b2 = consumption_complete(cp)
    debt_complete = [b1, b2]
    isapprox((c_bar + b2 - cp.y[2] - debt_complete' * (cp.β * cp.P)[2, :])[1], 0)

.. code-block:: julia
    :class: test

    @testset "Budget Constraint Tests" begin
        # assert that the object above is always true
        @test isapprox((c_bar + b2 - cp.y[2] - debt_complete' * (cp.β * cp.P)[2, :])[1], 0)
        @test c_bar ≈ 1.7241558441558444 # default example invariance
        @test debt_complete[2] ≈ 2.188311688311688 # one of the other objects
    end

Below, we'll take the outcomes produced by this code -- in particular the implied
consumption and debt paths -- and compare them with outcomes
from an incomplete markets model in the spirit of Hall :cite:`Hall1978` and Barro :cite:`Barro1979` (and also, for those who love history, Gallatin (1807) :cite:`Gallatin`)

Model 2 (One-Period Risk Free Debt Only)
========================================

This is a version of the original models of Hall (1978) and Barro (1979)
in which the decision maker's ability to substitute intertemporally is
constrained by his ability to buy or sell only one security, a risk-free
one-period bond bearing a constant gross interest rate that equals
:math:`\beta^{-1}`

Given an initial debt  :math:`b_0` at time :math:`0`, the
consumer faces a sequence of budget constraints

.. math::

    c_t + b_t = y_t + \beta b_{t+1}, \quad t \geq 0

where :math:`\beta` is the price at time :math:`t` of a risk-free claim
on one unit of time consumption at time :math:`t+1`

First-order conditions for the consumer's  problem are

.. math::

    \sum_{j} u'(c_{t+1,j}) P_{ij} = u'(c_{t, i})

For our assumed quadratic utility function this implies

.. math::
    :label: cs_8

    \sum_j c_{t+1,j} P_{ij} = c_{t,i}    ,

which is Hall's (1978) conclusion that consumption follows a random walk

As we saw in our first lecture on the :doc:`permanent income model <perm_income>`, this leads to

.. math::
    :label: cs_9

    b_t = \mathbb E_t \sum_{j=0}^\infty \beta^j y_{t+j} - (1 -\beta)^{-1} c_t

and

.. math::
    :label: cs_10

    c_t = (1-\beta)
        \left[
            \mathbb E_t \sum_{j=0}^\infty \beta^j y_{t+j} - b_t
        \right]   .

Equation :eq:`cs_10` expresses :math:`c_t` as a net interest rate factor :math:`1 - \beta` times the sum
of the expected present value of nonfinancial income :math:`\mathbb E_t \sum_{j=0}^\infty \beta^j y_{t+j}` and financial wealth :math:`-b_t`

Substituting :eq:`cs_10`  into the one-period budget constraint and rearranging leads to

.. math::
    :label: cs_11

    b_{t+1} - b_t
    = \beta^{-1} \left[ (1-\beta)
    \mathbb E_t \sum_{j=0}^\infty\beta^j y_{t+j} - y_t    \right]

Now let's do a useful calculation that will yield a convenient expression for the key term :math:`\mathbb E_t \sum_{j=0}^\infty\beta^j y_{t+j}` in our finite Markov chain setting

Define

.. math::

    v_t := \mathbb E_t \sum_{j=0}^\infty \beta^j y_{t+j}

In our finite Markov chain setting, :math:`v_t = v(1)` when :math:`s_t= \bar s_1` and :math:`v_t = v(2)` when :math:`s_t=\bar s_2`

Therefore, we can write

.. math::

    \begin{aligned}
        v(1) & = y(1) + \beta P_{11} v(1) + \beta P_{12} v(2)
        \\
        v(2) & = y(2) + \beta P_{21} v(1) + \beta P_{22} v(2)
    \end{aligned}

or

.. math::

    \vec v = \vec y + \beta P \vec v

where  :math:`\vec v =    \begin{bmatrix} v(1) \cr v(2) \end{bmatrix}` and  :math:`\vec y =  \begin{bmatrix} y(1) \cr y(2) \end{bmatrix}`

We can also write the last expression as

.. math::

    \vec v = (I - \beta P)^{-1} \vec y

In our finite Markov chain setting, from expression  :eq:`cs_10`,  consumption at date :math:`t` when debt is :math:`b_t` and the Markov state today is :math:`s_t = i` is evidently

.. math::
    :label: cs_12

    c(b_t, i) =  (1 - \beta) \left( [(I - \beta P)^{-1} \vec y]_i - b_t \right)

and the increment in debt is

.. math::
    :label: cs_13

    b_{t+1} - b_t = \beta^{-1} [ (1- \beta) v(i) - y(i) ]

Summary of Outcomes
-------------------

In contrast to outcomes in the complete markets model, in the incomplete
markets model

-  consumption drifts over time as a random walk; the level of
   consumption at time :math:`t` depends on the level of debt that the
   consumer brings into the period as well as the expected discounted
   present value of nonfinancial income at :math:`t`

-  the consumer's debt drifts upward over time in response to low
   realizations of nonfinancial income and drifts downward over time in
   response to high realizations of nonfinancial income

-  the drift over time in the consumer's debt and the dependence of
   current consumption on today's debt level account for the drift over
   time in consumption

The Incomplete Markets Model
-------------------------------------

The code above also contains a function called `consumption_incomplete()` that uses :eq:`cs_12` and :eq:`cs_13` to

*  simulate paths of :math:`y_t, c_t, b_{t+1}`

*  plot these against values of of :math:`\bar c, b(s_1), b(s_2)` found in a corresponding  complete markets economy

Let's try this, using the same parameters in both complete and incomplete markets economies

.. code-block:: julia

    Random.seed!(42)
    N_simul = 150
    cp = ConsumptionProblem()

    c_bar, b1, b2 = consumption_complete(cp)
    debt_complete = [b1, b2]

    c_path, debt_path, y_path, s_path = consumption_incomplete(cp, N_simul=N_simul)

    plt_cons = plot(title = "Consumption paths", xlabel = "Periods", ylim = [1.4,2.1])
    plot!(plt_cons, 1:N_simul, c_path, label = "incomplete market", lw = 2)
    plot!(plt_cons, 1:N_simul, fill(c_bar, N_simul), label = "complete market", lw = 2)
    plot!(plt_cons, 1:N_simul, y_path, label = "income", lw = 2, alpha = 0.6, linestyle = :dash)
    plot!(plt_cons, legend = :bottom)

    plt_debt = plot(title = "Debt paths", xlabel = "Periods")
    plot!(plt_debt, 1:N_simul, debt_path, label = "incomplete market")
    plot!(plt_debt, 1:N_simul, debt_complete[s_path], label = "complete market", lw = 2)
    plot!(plt_debt, 1:N_simul, y_path, label = "income", lw = 2, alpha = 0.6, linestyle = :dash)
    plot!(plt_debt, legend = :bottomleft)

    plot(plt_cons, plt_debt, layout = (1,2), size = (800, 400))

.. code-block:: julia
    :class: test

    @testset "First Plot Tests" begin
        @test c_path[4] ≈ 1.7436363636363656
        @test debt_path[2] ≈ 2.8376623376623398
        @test debt_complete ≈  [3.0, 2.188311688311688]
    end

In the graph on the left, for the same sample path of nonfinancial
income :math:`y_t`, notice that

*  consumption is constant when there are complete markets, but it takes a random walk in the incomplete markets version of the model

*  the consumer's debt oscillates between two values that are functions
   of the Markov state in the complete markets model, while the
   consumer's debt drifts in a "unit root" fashion in the incomplete
   markets economy

Using the Isomorphism
^^^^^^^^^^^^^^^^^^^^^

We can simply relabel variables to acquire tax-smoothing interpretations of our two models

.. code-block:: julia

    plt_tax = plot(title = "Tax collection paths", x_label = "Periods", ylim = [1.4,2.1])
    plot!(plt_tax, 1:N_simul, c_path, label = "incomplete market", lw = 2)
    plot!(plt_tax, 1:N_simul, fill(c_bar, N_simul), label = "complete market", lw = 2)
    plot!(plt_tax, 1:N_simul, y_path, label = "govt expenditures", alpha = .6, linestyle = :dash,
          lw = 2)

    plt_gov = plot(title = "Government assets paths", x_label = "Periods")
    plot!(plt_gov, 1:N_simul, debt_path, label = "incomplete market", lw = 2)
    plot!(plt_gov, 1:N_simul, debt_complete[s_path], label = "complete market", lw = 2)
    plot!(plt_gov, 1:N_simul, y_path, label = "govt expenditures", alpha = .6, linestyle = :dash,
          lw = 2)
    hline!(plt_gov, [0], linestyle = :dash, color = :black, lw = 2, label = "")
    plot(plt_tax, plt_gov, layout = (1,2), size = (800, 400))

.. code-block:: julia
  :class: test

  # This is the same plot as the above, simply relabeled.

Example: Tax Smoothing with Complete Markets
============================================

It is useful to focus on a simple tax-smoothing example with complete markets.

This example will illustrate how, in a complete markets model like that of Lucas and Stokey :cite:`LucasStokey1983`, the government purchases
insurance from the private sector.

    * Purchasing insurance  protects the government against the need to raise taxes too high or issue too much debt in the high government expenditure event.

We assume that government expenditures move between two values :math:`G_1 < G_2`, where Markov state :math:`1` means "peace" and Markov state :math:`2` means "war"

The government budget constraint in Markov state :math:`i` is

.. math::

    T_i + b_i = G_i + \sum_j Q_{ij} b_j

where

.. math::

    Q_{ij} = \beta P_{ij}

is the price of one unit of output next period in state :math:`j` when
today's Markov state is :math:`i` and :math:`b_i` is the government's
level of *assets* in Markov state :math:`i`

That is, :math:`b_i` is the amount  of the  one-period loans owned by the government that fall due at time :math:`t`

As above, we'll assume that the initial Markov state is state :math:`1`

In addition, to simplify our example, we'll set the government's initial
asset level to :math:`0`, so that :math:`b_1 =0`

Here's our code to compute a quantitative example with zero debt in peace time:

.. code-block:: julia

    # Parameters

    β = .96
    y = [1.0, 2.0]
    b0 = 0.0
    P = [0.8 0.2;
         0.4 0.6]

    cp = ConsumptionProblem(β, y, b0, P)
    Q = β * P
    N_simul = 150

    c_bar, b1, b2 = consumption_complete(cp)
    debt_complete = [b1, b2]

    println("P = $P")
    println("Q = $Q")
    println("Govt expenditures in peace and war = $y")
    println("Constant tax collections = $c_bar")
    println("Govt assets in two states = $debt_complete")

    msg = """
    Now let's check the government's budget constraint in peace and war.
    Our assumptions imply that the government always purchases 0 units of the
    Arrow peace security.
    """
    println(msg)

    AS1 = Q[1, 2] * b2
    println("Spending on Arrow war security in peace = $AS1")
    AS2 = Q[2, 2] * b2
    println("Spending on Arrow war security in war = $AS2")

    println("\n")
    println("Government tax collections plus asset levels in peace and war")
    TB1 = c_bar + b1
    println("T+b in peace = $TB1")
    TB2 = c_bar + b2
    println("T+b in war = $TB2")

    println("\n")
    println("Total government spending in peace and war")
    G1= y[1] + AS1
    G2 = y[2] + AS2
    println("total govt spending in peace = $G1")
    println("total govt spending in war = $G2")

    println("\n")
    println("Let's see ex post and ex ante returns on Arrow securities")

    Π = 1 ./ Q    # reciprocal(Q)
    exret = Π
    println("Ex post returns to purchase of Arrow securities = $exret")
    exant = Π .* P
    println("Ex ante returns to purchase of Arrow securities = $exant")

.. code-block:: julia
    :class: test

    @testset "Peace and War Debt Tests" begin
        @test c_bar ≈ 1.3116883116883118
        @test G2 ≈ 2.9350649350649354
        @test G1 ≈ 1.3116883116883118
        @test AS2 ≈ 0.9350649350649349
        @test AS1 ≈ 0.3116883116883117
    end

Explanation
-----------

In this example, the government always purchase :math:`0` units of the
Arrow security that pays off in peace time (Markov state :math:`1`)

But it purchases a positive amount of the security that pays off in war
time (Markov state :math:`2`)

We recommend plugging the quantities computed above into the government
budget constraints in the two Markov states and staring

This is an example in which the government purchases *insurance* against
the possibility that war breaks out or continues

*  the insurance does not pay off so long as peace continues

*  the insurance pays off when there is war

*Exercise:* try changing the Markov transition matrix so that

.. math::

    P = \begin{bmatrix}
            1 & 0 \\
           .2 & .8
        \end{bmatrix}

Also, start the system in Markov state :math:`2` (war) with initial
government assets :math:`- 10`, so that the government starts the
war in debt and :math:`b_2 = -10`

Linear State Space Version of Complete Markets Model
=====================================================

Now we'll use a setting like that in  :doc:`first lecture on the permanent income model <perm_income>`

In that model, there were

* incomplete markets: the consumer could trade only a single risk-free one-period bond bearing gross one-period risk-free interest rate equal to :math:`\beta^{-1}`

* the consumer's exogenous nonfinancial income was governed by a linear state space model driven by Gaussian shocks, the kind of model studied in an earlier lecture about :doc:`linear state space models <linear_models>`

We'll write down a complete markets counterpart of that model

So now we'll  suppose that nonfinancial income is governed by the state
space system

.. math::

    \begin{aligned}
         x_{t+1} & = A x_t + C w_{t+1} \cr
         y_t & = S_y x_t
    \end{aligned}

where :math:`x_t` is an :math:`n \times 1` vector and :math:`w_{t+1} \sim {\cal N}(0,I)` is IID over time

Again, as a counterpart of the Hall-Barro assumption that the risk-free
gross interest rate is :math:`\beta^{-1}`, we assume the scaled prices
of one-period ahead Arrow securities are

.. math::
    :label: cs_14

    p_{t+1}(x_{t+1} \,|\, x_t) = \beta \phi(x_{t+1} \,|\, A x_t, CC')

where :math:`\phi(\cdot \,|\, \mu, \Sigma)` is a multivariate Gaussian
distribution with mean vector :math:`\mu` and covariance matrix
:math:`\Sigma`

Let :math:`b(x_{t+1})` be a vector of state-contingent debt due at :math:`t+1`
as a function of the :math:`t+1` state :math:`x_{t+1}`.

Using the pricing function assumed in :eq:`cs_14`, the value at
:math:`t` of :math:`b(x_{t+1})` is

.. math::

    \beta \int b(x_{t+1}) \phi(x_{t+1} \,|\, A x_t, CC') d x_{t+1} = \beta  \mathbb E_t b_{t+1}

In the complete markets setting, the consumer faces a sequence of budget
constraints

.. math::

    c_t + b_t = y_t + \beta \mathbb E_t b_{t+1}, t \geq 0

We can solve the time :math:`t` budget constraint forward to obtain

.. math::

    b_t = \mathbb E_t  \sum_{j=0}^\infty \beta^j (y_{t+j} - c_{t+j} )

We assume as before that the consumer cares about the expected value
of

.. math::

    \sum_{t=0}^\infty \beta^t u(c_t), \quad 0 < \beta < 1

In the incomplete markets version of the model, we assumed that
:math:`u(c_t) = - (c_t -\gamma)^2`, so that the above utility functional
became

.. math::

    - \sum_{t=0}^\infty \beta^t ( c_t - \gamma)^2, \quad 0 < \beta < 1

But in the complete markets version, we can assume a more general form
of utility function that satisfies :math:`u' > 0` and :math:`u'' < 0`

The first-order condition for the consumer's problem with complete
markets and our assumption about Arrow securities prices is

.. math::

    u'(c_{t+1}) = u'(c_t) \quad \text{for all }  t\geq 0

which again implies :math:`c_t = \bar c` for some :math:`\bar c`

So it follows that

.. math::

    b_t = \mathbb E_t \sum_{j=0}^\infty \beta^j (y_{t+j} - \bar c)

or

.. math::
    :label: cs_15

    b_t = S_y (I - \beta A)^{-1} x_t - \frac{1}{1-\beta} \bar c

where the value of :math:`\bar c` satisfies

.. math::
    :label: cs_16

    \bar b_0 = S_y (I - \beta A)^{-1} x_0 - \frac{1}{1 - \beta } \bar c

where :math:`\bar b_0` is an initial level of the consumer's debt, specified
as a parameter of the problem

Thus, in the complete markets version of the consumption-smoothing
model, :math:`c_t = \bar c, \forall t \geq 0` is determined by :eq:`cs_16`
and the consumer's debt is a fixed function of
the state :math:`x_t` described by :eq:`cs_15`

Here's an example that shows how in this setting the availability of insurance against fluctuating nonfinancial income
allows the consumer completely to smooth consumption across time and across states of the world.

.. code-block:: julia
    :class: test

    Random.seed!(42);

.. code-block:: julia

    function complete_ss(β, b0, x0, A, C, S_y, T = 12)

        # Create a linear state space for simulation purposes
        # This adds "b" as a state to the linear state space system
        # so that setting the seed places shocks in same place for
        # both the complete and incomplete markets economy
        # Atilde = vcat(hcat(A, zeros(size(A,1), 1)),
        #               zeros(1, size(A,2) + 1))
        # Ctilde = vcat(C, zeros(1, 1))
        # S_ytilde = hcat(S_y, zeros(1, 1))

        lss = LSS(A, C, S_y, mu_0=x0)

        # Add extra state to initial condition
        # x0 = hcat(x0, 0)

        # Compute the (I - β*A)^{-1}
        rm = inv(I - β * A)

        # Constant level of consumption
        cbar = (1 - β) * (S_y * rm * x0 .- b0)
        c_hist = ones(T) * cbar[1]

        # Debt
        x_hist, y_hist = simulate(lss, T)
        b_hist = (S_y * rm * x_hist .- cbar[1] / (1.0 - β))

        return c_hist, vec(b_hist), vec(y_hist), x_hist
    end

    N_simul = 150

    # Define parameters
    α, ρ1, ρ2 = 10.0, 0.9, 0.0
    σ = 1.0
    # N_simul = 1
    # T = N_simul
    A = [1.0 0.0 0.0;
         α    ρ1  ρ2;
         0.0 1.0 0.0]
    C = [0.0, σ, 0.0]
    S_y = [1.0 1.0 0.0]
    β, b0 = 0.95, -10.0
    x0 = [1.0, α / (1 - ρ1), α / (1 - ρ1)]

    # Do simulation for complete markets
    out = complete_ss(β, b0, x0, A, C, S_y, 150)
    c_hist_com, b_hist_com, y_hist_com, x_hist_com = out

    # Consumption plots
    plt_cons = plot(title = "Cons and income", xlabel = "Periods", ylim = [-5.0, 110])
    plot!(plt_cons, 1:N_simul, c_hist_com, label = "consumption", lw = 2)
    plot!(plt_cons, 1:N_simul, y_hist_com, label = "income",
          lw = 2, alpha = 0.6, linestyle = :dash)

    # Debt plots
    plt_debt = plot(title = "Debt and income", xlabel = "Periods")
    plot!(plt_debt, 1:N_simul, b_hist_com, label = "debt", lw = 2)
    plot!(plt_debt, 1:N_simul, y_hist_com, label = "Income",
          lw = 2, alpha = 0.6, linestyle = :dash)
    hline!(plt_debt, [0], color = :black, linestyle = :dash, lw = 2, label = "")
    plot(plt_cons, plt_debt, layout = (1,2), size = (800, 400))
    plot!(legend = :bottomleft)

.. code-block:: julia
    :class: test

    @testset "Final Plot Tests" begin
        @test y_hist_com[3] ≈ 102.50832541673434
        @test c_hist_com[3] ≈ 101.50000000000003
        @test b_hist_com[3] ≈ 0.4022442533400863
        @test x_hist_com[3] ≈ 100.00000000000003
    end

Interpretation of Graph
-----------------------

In the above graph, please note that:

-  nonfinancial income fluctuates in a stationary manner

-  consumption is completely constant

-  the consumer's debt fluctuates in a stationary manner; in fact, in
   this case because nonfinancial income is a first-order
   autoregressive process, the consumer's debt is an exact affine function
   (meaning linear plus a constant) of the consumer's nonfinancial
   income

Incomplete Markets Version
--------------------------

The incomplete markets version of the model with nonfinancial income being governed by a linear state space system
is described in the first lecture on the :doc:`permanent income model <perm_income>` and the followup
lecture on  the :doc:`permanent income model <perm_income_cons>`

In that version, consumption follows a random walk and the consumer's debt follows a process with a unit root

We leave it to the reader to apply the usual isomorphism to deduce the corresponding implications for a tax-smoothing model like Barro's :cite:`Barro1979`

Government Manipulation of  Arrow Securities Prices
----------------------------------------------------

In :doc:`optimal taxation in an LQ economy<lqramsey>` and :doc:`recursive optimal taxation <opt_tax_recur>`, we study **complete-markets**
models in which the government recognizes that it can manipulate  Arrow securities prices

In :doc:`optimal taxation with incomplete markets <amss>`, we study an **incomplete-markets** model in which the government  manipulates asset prices
