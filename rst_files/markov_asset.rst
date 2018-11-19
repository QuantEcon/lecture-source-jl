.. _mass:

.. include:: /_static/includes/lecture_howto_jl_full.raw

.. highlight:: julia

*********************************************
:index:`Asset Pricing I: Finite State Models`
*********************************************

.. index::
    single: Models; Markov Asset Pricing

.. contents:: :depth: 2

.. epigraph::

    "A little knowledge of geometric series goes a long way" -- Robert E. Lucas, Jr.

.. epigraph::

    "Asset pricing is all about covariances" -- Lars Peter Hansen

Overview
========

.. index::
    single: Markov Asset Pricing; Overview

An asset is a claim on one or more future payoffs

The spot price of an asset depends primarily on

* the anticipated dynamics for the stream of income accruing to the owners

* attitudes to risk

* rates of time preference

In this lecture we consider some standard pricing models and dividend stream specifications

We study how prices and dividend-price ratios respond in these different scenarios

We also look at creating and pricing *derivative* assets by repackaging income streams

Key tools for the lecture are

* formulas for predicting future values of functions of a Markov state

* a formula for predicting the discounted sum of future values of a Markov state

:index:`Pricing Models`
=======================

.. index::
    single: Models; Pricing

In what follows let :math:`\{d_t\}_{t \geq 0}` be a stream of dividends

* A time-:math:`t` **cum-dividend** asset is a claim to the stream :math:`d_t, d_{t+1}, \ldots`

* A time-:math:`t` **ex-dividend** asset is a claim to the stream :math:`d_{t+1}, d_{t+2}, \ldots`

Let's look at some equations that we expect to hold for prices of assets under ex-dividend contracts
(we will consider cum-dividend pricing in the exercises)

Risk Neutral Pricing
--------------------

.. index::
    single: Pricing Models; Risk Neutral

Our first scenario is risk-neutral pricing

Let :math:`\beta = 1/(1+\rho)` be an intertemporal discount factor, where
:math:`\rho` is the rate at which agents discount the future

The basic risk-neutral asset pricing equation for pricing one unit of an ex-dividend asset is

.. _mass_pra:

.. math::
    :label: rnapex

    p_t = \beta {\mathbb E}_t [d_{t+1} + p_{t+1}]

This is a simple "cost equals expected benefit" relationship

Here :math:`{\mathbb E}_t [y]` denotes the best forecast of :math:`y`, conditioned on information available at time :math:`t`

Pricing with Random Discount Factor
-----------------------------------

.. index::
    single: Pricing Models; Risk Aversion

What happens if for some reason traders discount payouts differently depending on the state of the world?

Michael Harrison and David Kreps :cite:`HarrisonKreps1979` and Lars Peter Hansen
and Scott Richard :cite:`HansenRichard1987` showed that in quite general
settings the price of an ex-dividend asset obeys

.. math::
    :label: lteeqs0

    p_t = {\mathbb E}_t \left[ m_{t+1}  ( d_{t+1} + p_{t+1} ) \right]

for some  **stochastic discount factor** :math:`m_{t+1}`

The fixed discount factor :math:`\beta` in :eq:`rnapex` has been replaced by the random variable :math:`m_{t+1}`

The way anticipated future payoffs are evaluated can now depend on various random outcomes

One example of this idea is that assets that tend to have good payoffs in bad states of the world might be regarded as more valuable

This is because they pay well when the funds are more urgently needed

We give examples of how the stochastic discount factor has been modeled below

Asset Pricing and Covariances
-----------------------------

Recall that, from the definition of a conditional covariance :math:`{\rm cov}_t (x_{t+1}, y_{t+1})`, we have

.. math::
    :label: lteeqs101

    {\mathbb E}_t (x_{t+1} y_{t+1}) = {\rm cov}_t (x_{t+1}, y_{t+1}) + {\mathbb E}_t x_{t+1} {\mathbb E}_t y_{t+1}

If we apply this definition to the asset pricing equation :eq:`lteeqs0` we obtain

.. math::
    :label: lteeqs102

    p_t = {\mathbb E}_t m_{t+1} {\mathbb E}_t (d_{t+1} + p_{t+1}) + {\rm cov}_t (m_{t+1}, d_{t+1}+ p_{t+1})

It is useful to regard equation :eq:`lteeqs102`   as a generalization of equation :eq:`rnapex`

* In equation :eq:`rnapex`, the stochastic discount factor :math:`m_{t+1} = \beta`,  a constant

* In equation :eq:`rnapex`, the covariance term :math:`{\rm cov}_t (m_{t+1}, d_{t+1}+ p_{t+1})` is zero because :math:`m_{t+1} = \beta`

Equation :eq:`lteeqs102` asserts that the covariance of the stochastic discount factor with the one period payout :math:`d_{t+1} + p_{t+1}` is an important determinant of the price :math:`p_t`

We give examples of some models of stochastic discount factors that have been proposed later in this lecture and also in a :doc:`later lecture <lucas_model>`

The Price-Dividend Ratio
------------------------

Aside from prices, another quantity of interest is the **price-dividend ratio** :math:`v_t := p_t / d_t`

Let's write down an expression that this ratio should satisfy

We can divide both sides of :eq:`lteeqs0` by :math:`d_t` to get

.. math::
    :label: pdex

    v_t = {\mathbb E}_t \left[ m_{t+1} \frac{d_{t+1}}{d_t} (1 + v_{t+1}) \right]

Below we'll discuss the implication of this equation

Prices in the Risk Neutral Case
===============================

What can we say about price dynamics on the basis of the models described above?

The answer to this question depends on

#. the process we specify for dividends

#. the stochastic discount factor and how it correlates with dividends

For now let's focus on the risk neutral case, where the stochastic discount factor is constant, and study how prices depend on the dividend process

Example 1: Constant dividends
-----------------------------

The simplest case is risk neutral pricing in the face of a constant, non-random dividend stream :math:`d_t = d > 0`

Removing the expectation from :eq:`rnapex` and iterating forward gives

.. math::

    \begin{aligned}
        p_t & = \beta (d + p_{t+1})
            \\
            & = \beta (d + \beta(d + p_{t+2}))
            \\
            & \quad \vdots
            \\
            & = \beta (d + \beta d + \beta^2 d +  \cdots + \beta^{k-2} d + \beta^{k-1} p_{t+k})
    \end{aligned}

Unless prices explode in the future, this sequence converges to

.. math::
    :label: ddet

    \bar p := \frac{\beta d}{1-\beta}

This price is the equilibrium price in the constant dividend case

Indeed, simple algebra shows that setting :math:`p_t = \bar p` for all :math:`t`
satisfies the equilibrium condition :math:`p_t = \beta (d + p_{t+1})`

Example 2: Dividends with deterministic growth paths
----------------------------------------------------

Consider a growing, non-random dividend process :math:`d_{t+1} = g d_t`
where :math:`0 < g \beta < 1`

While prices are not usually constant when dividends grow over time, the price
dividend-ratio might be

If we guess this, substituting :math:`v_t = v` into :eq:`pdex` as well as our
other assumptions, we get :math:`v = \beta g (1 + v)`

Since :math:`\beta g < 1`, we have a unique positive solution:

.. math::

    v = \frac{\beta g}{1 - \beta g }

The price is then

.. math::

    p_t = \frac{\beta g}{1 - \beta g } d_t

If, in this example, we take :math:`g = 1+\kappa` and let
:math:`\rho := 1/\beta - 1`, then the price becomes

.. math::

    p_t = \frac{1 + \kappa}{ \rho - \kappa} d_t

This is called the *Gordon formula*

.. _mass_mg:

Example 3: Markov growth, risk neutral pricing
----------------------------------------------

Next we consider a dividend process

.. math::
    :label: mass_fmce

    d_{t+1} = g_{t+1} d_t

The stochastic growth factor :math:`\{g_t\}` is given by

.. math::

    g_t = g(X_t), \quad t = 1, 2, \ldots

where

#. :math:`\{X_t\}` is a finite Markov chain with state space :math:`S` and
   transition probabilities

.. math::

    P(x, y) := \mathbb P \{ X_{t+1} = y \,|\, X_t = x \}
    \qquad (x, y \in S)

#. :math:`g` is a given function on :math:`S` taking positive values

You can think of

* :math:`S` as :math:`n` possible "states of the world" and :math:`X_t` as the
  current state

* :math:`g` as a function that maps a given state :math:`X_t` into a growth
  factor :math:`g_t = g(X_t)` for the endowment

* :math:`\ln g_t = \ln (d_{t+1} / d_t)` is the growth rate of dividends

(For a refresher on notation and theory for finite Markov chains see :doc:`this lecture <finite_markov>`)

The next figure shows a simulation, where

* :math:`\{X_t\}` evolves as a discretized AR1 process produced using :ref:`Tauchen's method <mc_ex3>`

* :math:`g_t = \exp(X_t)`, so that :math:`\ln g_t = X_t` is the growth rate

Setup
-----

.. literalinclude:: /_static/includes/deps.jl

.. code-block:: julia
    :class: test

    using Test, Random

.. code-block:: julia

    using Parameters, Plots, QuantEcon
    gr(fmt = :png)

.. code-block:: julia
    :class: test

    Random.seed!(42);

.. code-block:: julia

    n = 25
    mc = tauchen(n, 0.96, 0.25)
    sim_length = 80

    x_series = simulate(mc, sim_length; init = round(Int, n / 2))
    g_series = exp.(x_series)
    d_series = cumprod(g_series) # assumes d_0 = 1

    series = [x_series g_series d_series log.(d_series)]
    labels = ["X_t" "g_t" "d_t" "ln(d_t)"]
    plot(series, layout = 4, labels = labels)

.. code-block:: julia
    :class: test

    @testset begin
        @test x_series[4] ≈ -0.669642857142857
        @test g_series[5] ≈ 0.4094841251523643
        @test d_series[9] ≈ 0.03514706070454392 # Near the inflection point.
        @test log.(d_series)[72] ≈ -29.24107142857142 # Something near the end.
    end

Pricing
^^^^^^^

To obtain asset prices in this setting, let's adapt our analysis from the case of deterministic growth

In that case we found that :math:`v` is constant

This encourages us to guess that, in the current case, :math:`v_t` is constant given the state :math:`X_t`

In other words, we are looking for a fixed function :math:`v` such that the price-dividend ratio satisfies  :math:`v_t = v(X_t)`

We can substitute this guess into :eq:`pdex` to get

.. math::

    v(X_t) = \beta {\mathbb E}_t [ g(X_{t+1}) (1 + v(X_{t+1})) ]

If we condition on :math:`X_t = x`, this becomes

.. math::

    v(x) = \beta \sum_{y \in S}  g(y) (1 + v(y)) P(x, y)

or

.. math::
    :label: pstack

    v(x) = \beta \sum_{y \in S}   K(x, y) (1 + v(y))
    \quad \text{where} \quad
    K(x, y) := g(y) P(x, y)

Suppose that there are :math:`n` possible states :math:`x_1, \ldots, x_n`

We can then think of :eq:`pstack` as :math:`n` stacked equations, one for each state, and write it in matrix form as

.. math::
    :label: vcumrn

    v = \beta K (\mathbb 1 + v)

Here

* :math:`v` is understood to be the column vector :math:`(v(x_1), \ldots, v(x_n))'`

* :math:`K` is the matrix :math:`(K(x_i, x_j))_{1 \leq i, j \leq n}`

* :math:`{\mathbb 1}` is a column vector of ones

When does :eq:`vcumrn` have a unique solution?

From the :ref:`Neumann series lemma <la_neumann>` and Gelfand's formula, this will be the case if :math:`\beta K` has spectral radius strictly less than one

In other words, we require that the eigenvalues of :math:`K`  be strictly less than :math:`\beta^{-1}` in modulus

The solution is then

.. math::
    :label: rned

    v = (I - \beta K)^{-1} \beta K{\mathbb 1}

Code
----

Let's calculate and plot the price-dividend ratio at a set of parameters

As before, we'll generate :math:`\{X_t\}`  as a :ref:`discretized AR1 process <mc_ex3>` and set :math:`g_t = \exp(X_t)`

Here's the code, including a test of the spectral radius condition

.. code-block:: julia

    n = 25  # size of state space
    β = 0.9
    mc = tauchen(n, 0.96, 0.02)

    K = mc.p .* exp.(mc.state_values)'

    v = (I - β * K) \  (β * K * ones(n, 1))

    plot(mc.state_values,
         v,
         lw = 2,
         ylabel = "price-dividend ratio",
         xlabel = "state",
         alpha = 0.7,
         label = "v")

.. code-block:: julia
    :class: test

    @testset begin
        @test v[2] == 3.4594684257743284
        @test v[1] == 3.2560393349907755
        @test v[5] ≈ 4.526909446326235
        @test K[8] ≈ 8.887213530262768e-10
    end

Why does the price-dividend ratio increase with the state?

The reason is that this Markov process is positively correlated, so high
current states suggest high future states

Moreover, dividend growth is increasing in the state

Anticipation of high future dividend growth leads to a high price-dividend ratio

Asset Prices under Risk Aversion
================================

Now let's turn to the case where agents are risk averse

We'll price several distinct assets, including

* The price of an endowment stream

* A consol (a type of bond issued by the UK government in the 19th century)

* Call options on a consol

Pricing a Lucas tree
--------------------

.. index::
    single: Finite Markov Asset Pricing; Lucas Tree

Let's start with a version of the celebrated asset pricing model of Robert E. Lucas, Jr. :cite:`Lucas1978`

As in :cite:`Lucas1978`, suppose that the stochastic discount factor takes the form

.. math::
    :label: lucsdf

    m_{t+1} = \beta \frac{u'(c_{t+1})}{u'(c_t)}

where :math:`u` is a concave utility function and :math:`c_t` is time :math:`t` consumption of a representative consumer

(A derivation of this expression is given in a :doc:`later lecture <lucas_model>`)

Assume the existence of an endowment that follows :eq:`mass_fmce`

The asset being priced is a claim on the endowment process

Following :cite:`Lucas1978`, suppose further that in equilibrium, consumption
is equal to the endowment, so that :math:`d_t = c_t` for all :math:`t`

For utility, we'll assume the **constant relative risk aversion** (CRRA)
specification

.. math::
    :label: eqCRRA

    u(c) = \frac{c^{1-\gamma}}{1 - \gamma} \ {\rm with} \ \gamma > 0

When :math:`\gamma =1` we let :math:`u(c) = \ln c`

Inserting the CRRA specification into :eq:`lucsdf` and using :math:`c_t = d_t` gives

.. math::
    :label: lucsdf2

    m_{t+1}
    = \beta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma}
    = \beta g_{t+1}^{-\gamma}

Substituting this into :eq:`pdex` gives the price-dividend ratio
formula

.. math::

    v(X_t)
    = \beta {\mathbb E}_t
    \left[
        g(X_{t+1})^{1-\gamma} (1 + v(X_{t+1}) )
    \right]

Conditioning on :math:`X_t = x`, we can write this as

.. math::

    v(x)
    = \beta \sum_{y \in S} g(y)^{1-\gamma} (1 + v(y) ) P(x, y)

If we let

.. math::

    J(x, y) := g(y)^{1-\gamma}  P(x, y)

then we can rewrite in vector form as

.. math::

    v = \beta J ({\mathbb 1} + v )

Assuming that the spectral radius of :math:`J` is strictly less than :math:`\beta^{-1}`, this equation has the unique solution

.. math::
    :label: resolvent2

    v = (I - \beta J)^{-1} \beta  J {\mathbb 1}

We will define a function `tree_price` to solve for $v$ given parameters stored in
the `AssetPriceModel` objects

.. code-block:: julia

    # A default Markov chain for the state process
    ρ = 0.9
    σ = 0.02
    n = 25
    default_mc = tauchen(n, ρ, σ)

    AssetPriceModel = @with_kw (β = 0.96,
                                γ = 2.0,
                                mc = default_mc,
                                n = size(mc.p)[1],
                                g = exp)

    # test stability of matrix Q
    function test_stability(ap, Q)
        sr = maximum(abs, eigvals(Q))
        if sr ≥ 1 / ap.β
            msg = "Spectral radius condition failed with radius = $sr"
            throw(ArgumentError(msg))
        end
    end

    # price/dividend ratio of the Lucas tree
    function tree_price(ap; γ = ap.γ)
        # Simplify names, set up matrices
        @unpack β, mc = ap
        P, y = mc.p, mc.state_values
        y = reshape(y, 1, ap.n)
        J = P .* ap.g.(y).^(1 - γ)

        # Make sure that a unique solution exists
        test_stability(ap, J)

        # Compute v
        v = (I - β * J) \ sum(β * J, dims = 2)

        return v
    end

Here's a plot of :math:`v` as a function of the state for several values of :math:`\gamma`,
with a positively correlated Markov process and :math:`g(x) = \exp(x)`

.. code-block:: julia

    γs = [1.2, 1.4, 1.6, 1.8, 2.0]
    ap = AssetPriceModel()
    states = ap.mc.state_values

    lines = []
    labels = []

    for γ in γs
        v = tree_price(ap, γ = γ)
        label = "gamma = $γ"
        push!(labels, label)
        push!(lines, v)
    end

    plot(lines,
         labels = reshape(labels, 1, length(labels)),
         title = "Price-dividend ratio as a function of the state",
         ylabel = "price-dividend ratio",
         xlabel = "state")

.. code-block:: julia
    :class: test

    @testset begin
        @test lines[2][4] ≈ 33.36574362637905
        @test lines[3][12] ≈ 28.52560591264372
        @test lines[4][18] ≈ 22.38597470787489
        @test lines[5][24] ≈ 15.81947255704859
    end

Notice that :math:`v` is decreasing in each case

This is because, with a positively correlated state process, higher states suggest higher future consumption growth

In the stochastic discount factor :eq:`lucsdf2`, higher growth decreases the
discount factor, lowering the weight placed on future returns

Special cases
^^^^^^^^^^^^^

In the special case :math:`\gamma =1`, we have :math:`J = P`

Recalling that :math:`P^i {\mathbb 1} = {\mathbb 1}` for all :math:`i` and applying :ref:`Neumann's geometric series lemma <la_neumann>`, we are led to

.. math::

    v = \beta(I-\beta P)^{-1} {\mathbb 1}
    = \beta \sum_{i=0}^{\infty} \beta^i P^i {\mathbb 1}
    = \beta \frac{1}{1 - \beta} {\mathbb 1}

Thus, with log preferences, the price-dividend ratio for a Lucas tree is constant

Alternatively, if :math:`\gamma = 0`, then :math:`J = K` and we recover the
risk neutral solution :eq:`rned`

This is as expected, since :math:`\gamma = 0` implies :math:`u(c) = c` (and hence agents are risk neutral)

A Risk-Free Consol
------------------

Consider the same pure exchange representative agent economy

A risk-free consol promises to pay a constant amount  :math:`\zeta> 0` each period

Recycling notation, let :math:`p_t` now be the price of an  ex-coupon claim to the consol

An ex-coupon claim to the consol entitles the owner at the end of period :math:`t` to

* :math:`\zeta` in period :math:`t+1`, plus

* the right to sell the claim for :math:`p_{t+1}` next period

The price satisfies :eq:`lteeqs0` with :math:`d_t = \zeta`, or

.. math::

    p_t = {\mathbb E}_t \left[ m_{t+1}  ( \zeta + p_{t+1} ) \right]

We maintain the stochastic discount factor :eq:`lucsdf2`, so this becomes

.. math::
    :label: consolguess1

    p_t
    = {\mathbb E}_t \left[ \beta g_{t+1}^{-\gamma}  ( \zeta + p_{t+1} ) \right]

Guessing a solution of the form :math:`p_t = p(X_t)` and conditioning on
:math:`X_t = x`, we get

.. math::

    p(x)
    = \beta \sum_{y \in S}  g(y)^{-\gamma} (\zeta + p(y)) P(x, y)

Letting :math:`M(x, y) = P(x, y) g(y)^{-\gamma}` and rewriting in vector notation
yields the solution

.. math::
    :label: consol_price

    p = (I - \beta M)^{-1} \beta M \zeta {\mathbb 1}

The above is implemented in the function `consol_price`

.. code-block:: julia

    function consol_price(ap, ζ)
        # Simplify names, set up matrices
        @unpack β, γ, mc, g, n = ap
        P, y = mc.p, mc.state_values
        y = reshape(y, 1, n)
        M = P .* g.(y).^(-γ)

        # Make sure that a unique solution exists
        test_stability(ap, M)

        # Compute price
        return (I - β * M) \ sum(β * ζ * M, dims = 2)
    end

Pricing an Option to Purchase the Consol
----------------------------------------

Let's now price options of varying maturity that give the right to purchase a consol at a price :math:`p_S`

An infinite horizon call option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We want to price an infinite horizon  option to purchase a consol at a price :math:`p_S`

The option entitles the owner at the beginning of a period either to

#. purchase the bond at price :math:`p_S` now, or

#. Not to exercise the option now but to retain the right to exercise it later

Thus, the owner either *exercises* the option now, or chooses *not to exercise* and wait until next period

This is termed an infinite-horizon *call option* with *strike price* :math:`p_S`

The owner of the option is entitled to purchase the consol at the price :math:`p_S` at the beginning of any period, after the coupon has been paid to the previous owner of the bond

The fundamentals of the economy are identical with the one above, including the stochastic discount factor and the process for consumption

Let :math:`w(X_t, p_S)` be the value of the option when the time :math:`t` growth state is known to be :math:`X_t` but *before* the owner has decided whether or not to exercise the option
at time :math:`t` (i.e., today)

Recalling that :math:`p(X_t)` is the value of the consol when the initial growth state is :math:`X_t`, the value of the option satisfies

.. math::

     w(X_t, p_S)
    = \max \left\{
        \beta \, {\mathbb E}_t \frac{u'(c_{t+1})}{u'(c_t)} w(X_{t+1}, p_S), \;
             p(X_t) - p_S
    \right\}

The first term on the right is the value of waiting, while the second is the value of exercising now

We can also write this as

.. math::
    :label: FEoption0

    w(x, p_S)
    = \max \left\{
        \beta \sum_{y \in S} P(x, y) g(y)^{-\gamma}
        w (y, p_S), \;
        p(x) - p_S
    \right\}

With :math:`M(x, y) = P(x, y) g(y)^{-\gamma}` and :math:`w` as the vector of
values :math:`(w(x_i), p_S)_{i = 1}^n`, we can express :eq:`FEoption0` as the nonlinear vector equation

.. math::
    :label: FEoption

    w = \max \{ \beta M w, \; p - p_S {\mathbb 1} \}

To solve :eq:`FEoption`, form the operator :math:`T` mapping vector :math:`w`
into vector :math:`Tw` via

.. math::

    T w
    = \max \{ \beta M w,\; p - p_S {\mathbb 1} \}

Start at some initial :math:`w` and iterate to convergence with :math:`T`

We can find the solution with the following function `call_option`

.. code-block:: julia

    # price of perpetual call on consol bond
    function call_option(ap, ζ, p_s, ϵ = 1e-7)

        # Simplify names, set up matrices
        @unpack β, γ, mc, g, n = ap
        P, y = mc.p, mc.state_values
        y = reshape(y, 1, n)
        M = P .* g.(y).^(-γ)

        # Make sure that a unique console price exists
        test_stability(ap, M)

        # Compute option price
        p = consol_price(ap, ζ)
        w = zeros(ap.n, 1)
        error = ϵ + 1
        while (error > ϵ)
            # Maximize across columns
            w_new = max.(β * M * w, p .- p_s)
            # Find maximal difference of each component and update
            error = maximum(abs, w - w_new)
            w = w_new
        end

        return w
    end

Here's a plot of :math:`w` compared to the consol price when :math:`P_S = 40`

.. code-block:: julia

    ap = AssetPriceModel(β=0.9)
    ζ = 1.0
    strike_price = 40.0

    x = ap.mc.state_values
    p = consol_price(ap, ζ)
    w = call_option(ap, ζ, strike_price)

    plot(x, p, color = "blue", lw = 2, xlabel = "state", label = "consol price")
    plot!(x, w, color = "green", lw = 2, label = "value of call option")

.. code-block:: julia
    :class: test

    @testset begin
        @test p[17] ≈ 9.302197030956606
        @test w[20] ≈ 0.46101660813737866
    end

In large states the value of the option is close to zero

This is despite the fact the Markov chain is irreducible and low states ---
where the consol prices is high --- will eventually be visited

The reason is that :math:`\beta=0.9`, so the future is discounted relatively rapidly

.. code-block:: julia
    :class: test

    @testset begin
        @test x[2] ≈ -0.126178653628809
        @test p[5] ≈ 52.85568616593254
    end

Risk Free Rates
---------------

Let's look at risk free interest rates over different periods

The one-period risk-free interest rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As before, the stochastic discount factor is :math:`m_{t+1} = \beta g_{t+1}^{-\gamma}`

It follows that the reciprocal :math:`R_t^{-1}` of the gross risk-free interest rate :math:`R_t` in state :math:`x` is

.. math::

    {\mathbb E}_t m_{t+1} = \beta \sum_{y \in S} P(x, y) g(y)^{-\gamma}

We can write this as

.. math::

    m_1 = \beta M {\mathbb 1}

where the :math:`i`-th  element of :math:`m_1` is the reciprocal of the one-period gross risk-free interest rate in state :math:`x_i`

Other terms
^^^^^^^^^^^

Let :math:`m_j` be an :math:`n \times 1` vector whose :math:`i` th component is the reciprocal of the :math:`j` -period gross risk-free interest rate in state :math:`x_i`

Then :math:`m_1 = \beta M`, and :math:`m_{j+1} = M m_j` for :math:`j \geq 1`

Exercises
=========

Exercise 1
----------

In the lecture, we considered **ex-dividend assets**

A **cum-dividend** asset is a claim to the stream :math:`d_t, d_{t+1}, \ldots`

Following :eq:`rnapex`, find the risk-neutral asset pricing equation for
one unit of a cum-dividend asset

With a constant, non-random dividend stream :math:`d_t = d > 0`, what is the equilibrium
price of a cum-dividend asset?

With a growing, non-random dividend process :math:`d_t = g d_t` where :math:`0 < g \beta < 1`,
what is the equilibrium price of a cum-dividend asset?

Exercise 2
----------

Consider the following primitives

.. code-block:: julia

    n = 5
    P = fill(0.0125, n, n) + (0.95 - 0.0125)I
    s = [1.05, 1.025, 1.0, 0.975, 0.95]
    γ = 2.0
    β = 0.94
    ζ = 1.0

Let :math:`g` be defined by :math:`g(x) = x`  (that is, :math:`g` is the identity map)

Compute the price of the Lucas tree

Do the same for

* the price of the risk-free consol when :math:`\zeta = 1`

* the call option on the consol when :math:`\zeta = 1` and :math:`p_S = 150.0`

Exercise 3
----------

Let's consider finite horizon call options, which are more common than the
infinite horizon variety

Finite horizon options obey functional equations closely related to :eq:`FEoption0`

A :math:`k` period option expires after :math:`k` periods

If we view today as date zero, a :math:`k` period option gives the owner the right to exercise the option to purchase the risk-free consol at the strike price :math:`p_S` at dates :math:`0, 1, \ldots , k-1`

The option expires at time :math:`k`

Thus, for :math:`k=1, 2, \ldots`, let :math:`w(x, k)` be the value of a :math:`k`-period option

It obeys

.. math::

    w(x, k)
    = \max \left\{
        \beta \sum_{y \in S} P(x, y) g(y)^{-\gamma}
        w (y, k-1), \;
        p(x) - p_S
    \right\}

where :math:`w(x, 0) = 0` for all :math:`x`

We can express the preceding as the sequence of nonlinear vector equations

.. math::

    w_k = \max \{ \beta M w_{k-1}, \; p - p_S {\mathbb 1} \}
      \quad k =1, 2, \ldots
      \quad \text{with } w_0 = 0

Write a function that computes :math:`w_k` for any given :math:`k`

Compute the value of the option with ``k = 5`` and ``k = 25`` using parameter values as in Exercise 1

Is one higher than the other?  Can you give intuition?

Solutions
=========

Exercise 2
----------

.. code-block:: julia

    n = 5
    P = fill(0.0125, n, n) + (0.95 - 0.0125)I
    s = [0.95, 0.975, 1.0, 1.025, 1.05]  # state values
    mc = MarkovChain(P, s)

    γ = 2.0
    β = 0.94
    ζ = 1.0
    p_s = 150.0

Next we'll create an instance of `AssetPriceModel` to feed into the functions.

.. code-block:: julia

    ap = AssetPriceModel(β = β, mc = mc, γ = γ, g = x -> x)

.. code-block:: julia

    v = tree_price(ap)
    println("Lucas Tree Prices: $v\n")

.. code-block:: julia
    :class: test

    @testset begin
        @test v[2] == 21.935706611219704
    end

.. code-block:: julia

    v_consol = consol_price(ap, 1.0)
    println("Consol Bond Prices: $(v_consol)\n")

.. code-block:: julia

    w = call_option(ap, ζ, p_s)

.. code-block:: julia
    :class: test

    @testset begin
        @test v_consol[1] ≈ 753.8710047641985
        @test w[2][1] ≈ 176.83933430191294
    end

Exercise 3
----------

Here's a suitable function:

.. code-block:: julia

    function finite_horizon_call_option(ap, ζ, p_s, k)

        # Simplify names, set up matrices
        @unpack β, γ, mc = ap
        P, y = mc.p, mc.state_values
        y = y'
        M = P .* ap.g.(y).^(- γ)

        # Make sure that a unique console price exists
        test_stability(ap, M)

        # Compute option price
        p = consol_price(ap, ζ)
        w = zeros(ap.n, 1)
        for i in 1:k
            # Maximize across columns
            w = max.(β * M * w, p .- p_s)
        end

        return w
    end

.. code-block:: julia
    :class: test

    @testset begin
        @test p[3] ≈ 70.00064625026326
        @test w[2] ≈ 176.83933430191294
    end

.. code-block:: julia

    lines = []
    labels = []
    for k in [5, 25]
        w = finite_horizon_call_option(ap, ζ, p_s, k)
        push!(lines, w)
        push!(labels, "k = $k")
    end
    plot(lines, labels = reshape(labels, 1, length(labels)))

.. code-block:: julia
    :class: test

    @testset begin
        @test lines[1][4] ≈ 29.859285398252347
        @test lines[2][2] ≈ 147.00074548801277
    end

Not surprisingly, the option has greater value with larger :math:`k`.
This is because the owner has a longer time horizon over which he or she
may exercise the option.
