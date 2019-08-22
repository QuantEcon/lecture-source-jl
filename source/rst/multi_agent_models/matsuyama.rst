.. _matsuyama:

.. include:: /_static/includes/header.raw

.. highlight:: julia

****************************************
Globalization and Cycles
****************************************

.. contents:: :depth: 2

Co-authored with Chase Coleman

Overview
=====================================

In this lecture, we review the paper `Globalization and Synchronization of Innovation Cycles <http://www.centreformacroeconomics.ac.uk/Discussion-Papers/2015/CFMDP2015-27-Paper.pdf>`__ by `Kiminori Matsuyama <http://faculty.wcas.northwestern.edu/~kmatsu/>`__, `Laura Gardini <http://www.mdef.it/index.php?id=32>`__ and `Iryna Sushko <http://irynasushko.altervista.org/>`__

This model helps us understand several interesting stylized facts about the world economy

One of these is synchronized business cycles across different countries

Most existing models that generate synchronized business cycles do so by assumption, since they tie output in each country to a common shock

They also fail to explain certain features of the data, such as the fact that the degree of synchronization tends to increase with trade ties

By contrast, in the model we consider in this lecture, synchronization is both endogenous and increasing with the extent of trade integration

In particular, as trade costs fall and international competition increases, innovation incentives become aligned and countries synchronize their innovation cycles

Background
-------------

The model builds on work by Judd :cite:`Judd1985`, Deneckner and Judd :cite:`Deneckere1992` and Helpman and Krugman :cite:`Helpman1985` by developing a two country model with trade and innovation

On the technical side, the paper introduces the concept of `coupled oscillators <https://en.wikipedia.org/wiki/Oscillation#Coupled_oscillations>`__ to economic modeling

As we will see, coupled oscillators arise endogenously within the model

Below we review the model and replicate some of the results on synchronization of innovation across countries

Key Ideas
==========================

It is helpful to begin with an overview of the mechanism

Innovation Cycles
---------------------

As discussed above, two countries produce and trade with each other

In each country, firms innovate, producing new varieties of goods and, in doing so, receiving temporary monopoly power

Imitators follow and, after one period of monopoly, what had previously been new varieties now enter competitive production

Firms have incentives to innovate and produce new goods when the mass of varieties of goods currently in production is relatively low

In addition, there are strategic complementarities in the timing of innovation

Firms have incentives to innovate in the same period, so as to avoid competing with substitutes that are competitively produced

This leads to temporal clustering in innovations in each country

After a burst of innovation, the mass of goods currently in production increases

However, goods also become obsolete, so that not all survive from period to period

This mechanism generates a cycle, where the mass of varieties increases through simultaneous innovation and then falls through obsolescence

Synchronization
---------------

In the absence of trade, the timing of innovation cycles in each country is decoupled

This will be the case when trade costs are prohibitively high

If trade costs fall, then goods produced in each country penetrate each other's markets

As illustrated below, this leads to synchonization of business cycles across the two countries

Model
=======

Let's write down the model more formally

(The treatment is relatively terse since full details can be found in `the original paper <http://www.centreformacroeconomics.ac.uk/Discussion-Papers/2015/CFMDP2015-27-Paper.pdf>`__)

Time is discrete with :math:`t = 0, 1, \dots`

There are two countries indexed by :math:`j` or :math:`k`

In each country, a representative household inelastically supplies :math:`L_j` units of labor at wage rate :math:`w_{j, t}`

Without loss of generality, it is assumed that :math:`L_{1} \geq L_{2}`

Households consume a single nontradeable final good which is produced competitively

Its production involves combining two types of tradeable intermediate inputs
via

.. math::

    Y_{k, t} = C_{k, t} = \left( \frac{X^o_{k, t}}{1 - \alpha} \right)^{1-\alpha} \left( \frac{X_{k, t}}{\alpha} \right)^{\alpha}

Here :math:`X^o_{k, t}` is a homogeneous input which can be produced from labor using a linear, one-for-one technology

It is freely tradeable, competitively supplied, and homogeneous across countries

By choosing the price of this good as numeraire and assuming both countries find it optimal to always produce the homogeneous good, we can set :math:`w_{1, t} = w_{2, t} = 1`

The good :math:`X_{k, t}` is a composite, built from many differentiated goods via

.. math::

    X_{k, t}^{1 - \frac{1}{\sigma}} = \int_{\Omega_t} \left[ x_{k, t}(\nu) \right]^{1 - \frac{1}{\sigma}} d \nu

Here :math:`x_{k, t}(\nu)` is the total amount of a differentiated good :math:`\nu \in \Omega_t` that is produced

The parameter :math:`\sigma > 1` is the direct partial elasticity of substitution between a pair of varieties and :math:`\Omega_t` is the set of varieties available in period
:math:`t`

We can split the varieties into those which are supplied competitively and those supplied monopolistically; that is, :math:`\Omega_t = \Omega_t^c + \Omega_t^m`

Prices
---------

Demand for differentiated inputs is

.. math::

    x_{k, t}(\nu) = \left( \frac{p_{k, t}(\nu)}{P_{k, t}} \right)^{-\sigma} \frac{\alpha  L_k}{P_{k, t}}

Here

* :math:`p_{k, t}(\nu)` is the price of the variety :math:`\nu` and

* :math:`P_{k, t}` is the price index for differentiated inputs in :math:`k`,
  defined by

.. math::

    \left[ P_{k, t} \right]^{1 - \sigma} = \int_{\Omega_t} [p_{k, t}(\nu) ]^{1-\sigma} d\nu

The price of a variety also depends on the origin, :math:`j`, and destination, :math:`k`, of the goods because shipping
varieties between countries incurs an iceberg trade cost
:math:`\tau_{j,k}`

Thus the effective price in country :math:`k` of a variety :math:`\nu` produced in country :math:`j` becomes :math:`p_{k, t}(\nu) = \tau_{j,k} \, p_{j, t}(\nu)`

Using these expressions, we can derive the total demand for each variety,
which is

.. math::

    D_{j, t}(\nu) = \sum_k \tau_{j, k} x_{k, t}(\nu) = \alpha A_{j, t}(p_{j, t}(\nu))^{-\sigma}

where

.. math::

    A_{j, t} := \sum_k \frac{\rho_{j, k}  L_{k}}{(P_{k, t})^{1 - \sigma}}
    \quad \text{and} \quad
    \rho_{j, k} = (\tau_{j, k})^{1 - \sigma} \leq 1

It is assumed that :math:`\tau_{1,1} = \tau_{2,2} = 1` and :math:`\tau_{1,2} = \tau_{2,1} = \tau` for some :math:`\tau > 1`, so that

.. math::

    \rho_{1,2} = \rho_{2,1} = \rho := \tau^{1 - \sigma} < 1

The value :math:`\rho \in [0, 1)` is a proxy for the degree of globalization

Producing one unit of each differentiated variety requires :math:`\psi` units of labor, so the marginal cost is equal to :math:`\psi` for :math:`\nu \in \Omega_{j, t}`

Additionally, all competitive varieties will have the same price (because of equal marginal cost), which means that, for all :math:`\nu \in \Omega^c`,

.. math::

    p_{j, t}(\nu) = p_{j, t}^c := \psi
    \quad \text{and} \quad
    D_{j, t} = y_{j, t}^c := \alpha A_{j, t} (p_{j, t}^c)^{-\sigma}

Monopolists will have the same marked-up price, so, for all :math:`\nu \in \Omega^m` ,

.. math::

    p_{j, t}(\nu) = p_{j, t}^m := \frac{\psi }{1 - \frac{1}{\sigma}}
    \quad \text{and} \quad
    D_{j, t} = y_{j, t}^m  := \alpha A_{j, t} (p_{j, t}^m)^{-\sigma}

Define

.. math::

    \theta
    := \frac{p_{j, t}^c}{p_{j, t}^m} \frac{y_{j, t}^c}{y_{j, t}^m}
    = \left(1 - \frac{1}{\sigma} \right)^{1-\sigma}

Using the preceding definitions and some algebra, the price indices can now be rewritten as

.. math::

    \left(\frac{P_{k,t}}{\psi}\right)^{1-\sigma} = M_{k,t}  + \rho M_{j,t}
    \quad \text{where} \quad
    M_{j,t} := N_{j,t}^c + \frac{N_{j,t}^m}{ \theta}

The symbols :math:`N_{j, t}^c` and :math:`N_{j, t}^m` will denote the measures of :math:`\Omega^c` and :math:`\Omega^m` respectively

New Varieties
--------------

To introduce a new variety, a firm must hire :math:`f` units of labor per variety in each country

Monopolist profits must be less than or equal to zero in expectation, so

.. math::

    N_{j,t}^m \geq 0, \quad
    \pi_{j, t}^m := (p_{j, t}^m - \psi) y_{j, t}^m - f \leq 0
    \quad \text{and} \quad
    \pi_{j, t}^m N_{j,t}^m = 0

With further manipulations, this becomes

.. math::

    N_{j,t}^m = \theta(M_{j,t} - N_{j,t}^c) \geq 0,
    \quad
    \frac{1}{\sigma}
    \left[
        \frac{\alpha L_j}{\theta(M_{j,t} + \rho M_{k,t})}
        +
        \frac{\alpha L_k}{\theta(M_{j,t} + M_{k,t} / \rho)}
    \right]
    \leq f

Law of Motion
---------------

With :math:`\delta` as the exogenous probability of a variety becoming obsolete,
the dynamic equation for the measure of firms becomes

.. math::

    N_{j, t+1}^c = \delta (N_{j, t}^c + N_{j, t}^m) = \delta (N_{j, t}^c + \theta(M_{j, t} - N_{j, t}^c))

We will work with a normalized measure of varieties

.. math::

    n_{j, t} := \frac{\theta \sigma f N_{j, t}^c}{\alpha (L_1 + L_2)},
    \quad
    i_{j, t} := \frac{\theta \sigma f N_{j, t}^m}{\alpha (L_1 + L_2)},
    \quad
    m_{j, t} := \frac{\theta \sigma f M_{j, t}}{\alpha (L_1 + L_2)} = n_{j, t} + \frac{i_{j, t}}{\theta}

We also use :math:`s_j := \frac{L_j}{L_1 + L_2}` to be the share of labor employed in country :math:`j`

We can use these definitions and the preceding expressions to obtain a law of
motion for :math:`n_t := (n_{1, t}, n_{2, t})`

In particular, given an initial condition, :math:`n_0 = (n_{1, 0}, n_{2, 0}) \in \mathbb{R}_{+}^{2}`, the equilibrium trajectory, :math:`\{ n_t \}_{t=0}^{\infty} = \{ (n_{1, t}, n_{2, t}) \}_{t=0}^{\infty}`, is obtained by iterating on :math:`n_{t+1} = F(n_t)` where :math:`F : \mathbb{R}_{+}^{2} \rightarrow \mathbb{R}_{+}^{2}` is given by

.. math::

    \begin{aligned}
      F(n_t)
      &=
      \begin{cases}
          \big( \delta (\theta s_1(\rho) + (1-\theta) n_{1, t}), \delta (\theta s_2(\rho) + (1-\theta) n_{2, t}) \big) \; & \text{for } n_t \in D_{LL} \\
          \big( \delta n_{1, t}, \delta n_{2, t} \big) \; &\text{for } n_t \in D_{HH}  \\
          \big( \delta n_{1, t}, \delta (\theta h_2(n_{1, t}) + (1-\theta) n_{2, t}) \big) &\text{for } n_t \in D_{HL}  \\
          \big( \delta (\theta h_1(n_{2, t}) + (1-\theta) n_{1, t}, \delta n_{2, t}) \big) &\text{for } n_t \in D_{LH}
      \end{cases}
    \end{aligned}

Here

.. math::

    \begin{aligned}
          D_{LL} & := \{ (n_1, n_2) \in \mathbb{R}_{+}^{2} | n_j \leq s_j(\rho) \} \\
          D_{HH} & := \{ (n_1, n_2) \in \mathbb{R}_{+}^{2} | n_j \geq h_j(\rho) \} \\
          D_{HL} & :=  \{ (n_1, n_2) \in \mathbb{R}_{+}^{2} | n_1 \geq s_1(\rho) \text{ and } n_2 \leq h_2(n_1) \} \\
          D_{LH} & :=  \{ (n_1, n_2) \in \mathbb{R}_{+}^{2} | n_1 \leq h_1(n_2) \text{ and } n_2 \geq s_2(\rho) \}
    \end{aligned}

while

.. math::

    s_1(\rho) = 1 - s_2(\rho)
    = \min \left\{ \frac{s_1 - \rho s_2}{1 - \rho}, 1 \right\}

and :math:`h_j(n_k)` is defined implicitly by the equation

.. math::

    1 = \frac{s_j}{h_j(n_k) + \rho n_k} + \frac{s_k}{h_j(n_k) + n_k / \rho}

Rewriting the equation above gives us a quadratic equation in terms of :math:`h_j(n_k)`

Since we know :math:`h_j(n_k) > 0` then we can just solve the quadratic equation and return the positive root

This gives us

.. math::

    h_j(n_k)^2 + \left( (\rho + \frac{1}{\rho}) n_k - s_j - s_k \right) h_j(n_k) + (n_k^2 - \frac{s_j n_k}{\rho} - s_k n_k \rho) = 0

Simulation
===============

Let's try simulating some of these trajectories

We will focus in particular on whether or not innovation cycles synchronize
across the two countries

As we will see, this depends on initial conditions

For some parameterizations, synchronization will occur for "most" initial conditions, while for others synchronization will be rare

Here's the main body of code

Setup
-----

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

    using LinearAlgebra, Statistics, Compat 
    using Plots, Parameters
    gr(fmt = :png);

.. code-block:: julia

    function h_j(j, nk, s1, s2, θ, δ, ρ)
        # Find out who's h we are evaluating
        if j == 1
            sj = s1
            sk = s2
        else
            sj = s2
            sk = s1
        end

        # Coefficients on the quadratic a x^2 + b x + c = 0
        a = 1.0
        b = ((ρ + 1 / ρ) * nk - sj - sk)
        c = (nk * nk - (sj * nk) / ρ - sk * ρ * nk)

        # Positive solution of quadratic form
        root = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

        return root
    end

    DLL(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ) =
        (n1 ≤ s1_ρ) && (n2 ≤ s2_ρ)

    DHH(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ) =
        (n1 ≥ h_j(1, n2, s1, s2, θ, δ, ρ)) && (n2 ≥ h_j(2, n1, s1, s2, θ, δ, ρ))

    DHL(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ) =
        (n1 ≥ s1_ρ) && (n2 ≤ h_j(2, n1, s1, s2, θ, δ, ρ))

    DLH(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ) =
        (n1 ≤ h_j(1, n2, s1, s2, θ, δ, ρ)) && (n2 ≥ s2_ρ)

    function one_step(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
        # Depending on where we are, evaluate the right branch
        if DLL(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
            n1_tp1 = δ * (θ * s1_ρ + (1 - θ) * n1)
            n2_tp1 = δ * (θ * s2_ρ + (1 - θ) * n2)
        elseif DHH(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
            n1_tp1 = δ * n1
            n2_tp1 = δ * n2
        elseif DHL(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
            n1_tp1 = δ * n1
            n2_tp1 = δ * (θ * h_j(2, n1, s1, s2, θ, δ, ρ) + (1 - θ) * n2)
        elseif DLH(n1, n2, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
            n1_tp1 = δ * (θ * h_j(1, n2, s1, s2, θ, δ, ρ) + (1 - θ) * n1)
            n2_tp1 = δ * n2
        end

        return n1_tp1, n2_tp1
    end

    new_n1n2(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ) =
        one_step(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)

    function pers_till_sync(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ,
                            maxiter, npers)

        # Initialize the status of synchronization
        synchronized = false
        pers_2_sync = maxiter
        iters = 0

        nsync = 0

        while (~synchronized) && (iters < maxiter)
            # Increment the number of iterations and get next values
            iters += 1

            n1_t, n2_t = new_n1n2(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)

            # Check whether same in this period
            if abs(n1_t - n2_t) < 1e-8
                nsync += 1
            # If not, then reset the nsync counter
            else
                nsync = 0
            end

            # If we have been in sync for npers then stop and countries
            # became synchronized nsync periods ago
            if nsync > npers
                synchronized = true
                pers_2_sync = iters - nsync
            end
            n1_0, n2_0 = n1_t, n2_t
        end
        return synchronized, pers_2_sync
    end

    function create_attraction_basis(s1_ρ, s2_ρ, s1, s2, θ, δ, ρ,
                                     maxiter, npers, npts)
        # Create unit range with npts
        synchronized, pers_2_sync = false, 0
        unit_range = range(0.0,  1.0, length = npts)

        # Allocate space to store time to sync
        time_2_sync = zeros(npts, npts)
        # Iterate over initial conditions
        for (i, n1_0) in enumerate(unit_range)
            for (j, n2_0) in enumerate(unit_range)
                synchronized, pers_2_sync = pers_till_sync(n1_0, n2_0, s1_ρ, s2_ρ,
                                                           s1, s2, θ, δ, ρ,
                                                           maxiter, npers)
                time_2_sync[i, j] = pers_2_sync
            end
        end
        return time_2_sync
    end

    # model
    function MSGSync(s1 = 0.5, θ = 2.5, δ = 0.7, ρ = 0.2)
        # Store other cutoffs and parameters we use
        s2 = 1 - s1
        s1_ρ = min((s1 - ρ * s2) / (1 - ρ), 1)
        s2_ρ = 1 - s1_ρ

        return (s1 = s1, s2 = s2, s1_ρ = s1_ρ, s2_ρ = s2_ρ, θ = θ, δ = δ, ρ = ρ)
    end

    function simulate_n(model, n1_0, n2_0, T)
        # Unpack parameters
        @unpack s1, s2, θ, δ, ρ, s1_ρ, s2_ρ = model

        # Allocate space
        n1 = zeros(T)
        n2 = zeros(T)

        # Simulate for T periods
        for t in 1:T
            # Get next values
            n1[t], n2[t] = n1_0, n2_0
            n1_0, n2_0 = new_n1n2(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2, θ, δ, ρ)
        end

        return n1, n2
    end

    function pers_till_sync(model, n1_0, n2_0,
                            maxiter = 500, npers = 3)
        # Unpack parameters
        @unpack s1, s2, θ, δ, ρ, s1_ρ, s2_ρ = model

        return pers_till_sync(n1_0, n2_0, s1_ρ, s2_ρ, s1, s2,
                            θ, δ, ρ, maxiter, npers)
    end

    function create_attraction_basis(model;
                                     maxiter = 250,
                                     npers = 3,
                                     npts = 50)
        # Unpack parameters
        @unpack s1, s2, θ, δ, ρ, s1_ρ, s2_ρ = model

        ab = create_attraction_basis(s1_ρ, s2_ρ, s1, s2, θ, δ,
                                     ρ, maxiter, npers, npts)
        return ab
    end

Time Series of Firm Measures
----------------------------

We write a short function below that exploits the preceding code and plots two time series

Each time series gives the dynamics for the two countries

The time series share parameters but differ in their initial condition

Here's the function

.. code-block:: julia

    function plot_timeseries(n1_0, n2_0, s1 = 0.5, θ = 2.5, δ = 0.7, ρ = 0.2)
        model = MSGSync(s1, θ, δ, ρ)
        n1, n2 = simulate_n(model, n1_0, n2_0, 25)
        return [n1 n2]
    end

    # Create figures
    data_ns = plot_timeseries(0.15, 0.35)
    data_s = plot_timeseries(0.4, 0.3)

    plot(data_ns, title = "Not Synchronized", legend = false)

.. code-block:: julia

    plot(data_s, title = "Synchronized", legend = false)

In the first case, innovation in the two countries does not synchronize

In the second case different initial conditions are chosen, and the cycles
become synchronized

Basin of Attraction
-----------------------

Next let's study the initial conditions that lead to synchronized cycles more
systematically

We generate time series from a large collection of different initial
conditions and mark those conditions with different colors according to
whether synchronization occurs or not

The next display shows exactly this for four different parameterizations (one
for each subfigure)

Dark colors indicate synchronization, while light colors indicate failure to synchronize

.. _matsrep:

.. figure:: /_static/figures/matsuyama_14.png
    :scale: 60%

As you can see, larger values of :math:`\rho` translate to more synchronization

You are asked to replicate this figure in the exercises

Exercises
==============

Exercise 1
------------

Replicate the figure :ref:`shown above <matsrep>` by coloring initial conditions according to whether or not synchronization occurs from those conditions

Solutions
==========

Exercise 1
----------

.. code-block:: julia

    function plot_attraction_basis(s1 = 0.5, θ = 2.5, δ = 0.7, ρ = 0.2; npts = 250)
        # Create attraction basis
        unitrange = range(0,  1, length = npts)
        model = MSGSync(s1, θ, δ, ρ)
        ab = create_attraction_basis(model,npts=npts)
        plt = Plots.heatmap(ab, legend = false)
    end

.. code-block:: julia

    params = [[0.5, 2.5, 0.7, 0.2],
              [0.5, 2.5, 0.7, 0.4],
              [0.5, 2.5, 0.7, 0.6],
              [0.5, 2.5, 0.7, 0.8]]

    plots = (plot_attraction_basis(p...) for p in params)
    plot(plots..., size = (1000, 1000))

.. code-block:: julia
    :class: test

    function plot_attraction_basis(s1 = 0.5,
                                θ = 2.5,
                                δ = 0.7,
                                ρ = 0.2;
                                npts = 250)
        # Create attraction basis
        unitrange = range(0,  1, length = npts)
        model = MSGSync(s1, θ, δ, ρ)
        ab = create_attraction_basis(model, npts = npts)
    end

    abvec = [plot_attraction_basis(p...) for p in params]

    @testset begin
        @test abvec[1][5] ≈ 194.0
        @test abvec[2][17] ≈ 102.0
        @test abvec[3][50] ≈ 76.0
        @test abvec[4][end-1] ≈ 86.0
    end
