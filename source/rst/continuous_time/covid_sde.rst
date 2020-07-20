.. _covid_sde:

.. include:: /_static/includes/header.raw

.. highlight:: julia

***************************************************************************
:index:`Modeling Shocks in COVID 19 with Stochastic Differential Equations`
****************************************************************************

.. contents:: :depth: 2

Overview
=============

Coauthored with Chris Rackauckas

This lecture continues the analyzing of the COVID-19 pandemic established in :doc:`this lecture <seir_model>`.


As before, the model is inspired by 
*  Notes from `Andrew Atkeson <https://sites.google.com/site/andyatkeson/>`__ and `NBER Working Paper No. 26867 <https://www.nber.org/papers/w26867>`__
* `Estimating and Forecasting Disease Scenarios for COVID-19 with an SIR Model <https://www.nber.org/papers/w27335>`__ by Andrew Atkeson, Karen Kopecky and Tao Zha
* `Estimating and Simulating a SIRD Model of COVID-19 for Many Countries, States, and Cities <https://www.nber.org/papers/w27128>`__ by Jesús Fernández-Villaverde and Charles I. Jones
* Further variations on the classic SIR model in Julia  `here <https://github.com/epirecipes/sir-julia>`__.


Here we extend the model to include policy-relevant aggregate shocks, and
examine the three main techniques for including stochasticity to continuous-time models:

* Brownian Motion:  A diffusion process with  stochastic, continuous paths.  The prototypical  Stochastic Differential Equation (SDE) with additive noise.
* Pure-Jump Processes: A variable that jumps between a discrete number of values, typically with a Poisson arrival rate.
* Jump-Diffusion Process: A stochastic process that contains both a diffusion term and arrival rates of discrete jumps.

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

    using OrdinaryDiffEq, StochasticDiffEq, DiffEqJump
    using Parameters, StaticArrays, Plots, RecursiveArrayTools


The Basic SIR/SIRD Model
=========================

To demonstrate another common `compartmentalized model <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#Elaborations_on_the_basic_SIR_model>`__ we will change the SIER model
to remove the exposed state, and more carefully manage the death state, D.

All individuals in the population are assumed to be in one of these four states.

The states are: susceptible (S), infected (I), resistant (R), or dead (D).


Comments:

* Unlike the previous SIER model, the R state is only for those recovered, alive, and currently resistant.  

* As before, those who have recovered are assumed to have acquired immunity and become resistant.

* Later, we could consider transitions from R to S if resistance is not permanent.

Transition Rates
-------------------------------

We assume that you have studied :doc:`the previous lecture <seir_model>`, and will provide a more concise development of the model.

* We still ignore birth and non-covid death during our time horizon, and assume a large, constant, number of individuals of size :math:`N` throughout
* :math:`\beta(t)` is called the *transmission rate* or *effective contact rate* (the rate at which individuals bump into others and expose them to the virus)
* :math:`\gamma` is called the *resolution rate* (the rate at which infected people recover or die)
* :math:`\delta(t) \in [0, 1]` is the *death probability*
* As before, we re-parameterize as :math:`R_0(t) := \beta(t) / \gamma`, where :math:`R_0` has previous interpretation
* We jump directly to the equations in :math:`s, i, r, d` already normalized by :math:`N`

Summarizing these equations,

.. math::
   \begin{aligned}
        d s  & = - \gamma \, R_0 \, s \,  i \, dt
        \\
         d i  & = \gamma \, R_0 \, s \,  i  - \gamma i \, dt
        \\
        d r  & = (1-\delta) \gamma  i \, dt 
        \\
        d d  & = \delta \gamma  i \, dt
        \\         
   \end{aligned}
   :label: SIRD_system



Introduction to SDEs: Aggregate Shocks to Transmission Rates
==============================================================

We will start by extending our model to include randomness in :math:`R_0(t)`, which makes it a system of Stochastic Differential Equations (SDEs).

Continuous-Time Stochastic Processes
-----------------------------------

In continuous-time, there is an important distinction between randomness that leads to continuous paths vs. those which may have jumps (which are almost surely right-continuous).  The most tractable of these includes the theory of `Levy Processes <https://en.wikipedia.org/wiki/L%C3%A9vy_process>`_.

.. **TBD:** Add definition of levy processes and the intuitive connection between stationary increments and independence of increments.

Among the appealing features of Levy Processes is that they fit well into the sorts of Markov modeling techniques that economists tend to use in discrete time.

Unlike in discrete-time, where a modeller has license to be creative, the rules of continuous-time stochastic processes are much stricter.  In practice, there are only two types of Levy Processes that can be used without careful measure theory.

#. `Weiner Processes <https://en.wikipedia.org/wiki/Wiener_process>`__ (as known as Brownian Motion) which leads to a diffusion equations, and is the only continuous-time Levy process with continuous paths
#. `Poisson Processes <https://en.wikipedia.org/wiki/Poisson_point_process>`__ with an arrival rate of jumps in the variable.

Every other Levy Process can be represented by these building blocks (e.g. a `Diffusion Process <https://en.wikipedia.org/wiki/Diffusion_process>`__ such as Geometric Brownian Motion is a transformation of a Weiner process, and a `jump diffusion <https://en.wikipedia.org/wiki/Jump_diffusion#In_economics_and_finance>`__ is a diffusion process with a Poisson arrival of jumps).

In this section, we will look at example of a diffusion process.


Laws of Motion
----------------------

We will assume that the transmission rate follows a process with a reversion to a value :math:`B(t)` which could conceivably be influenced by policy.  The intuition is that even if the targeted :math:`B(t)` was changed through social distancing/etc., lags in behavior and implementation would smooth out the transition, where :math:`\eta` governs the speed of :math:`R_0(t)` moves towards :math:`B(t)`.

.. math::
    \begin{aligned}
    \frac{d R_0}{d t} &= \eta (B - R_0)\\
    \end{aligned}
    :label: Rode


Finally, let :math:`m(t)` be the mortality rate, which we will leave constant for now, i.e. :math:`\frac{d m}{d t} = 0`.  The cumulative deaths can be integrated through the flow :math:`\gamma i` entering the "Removed" state and define the cumulative number of deaths as :math:`D(t)`.

.. math::
    \begin{aligned}\\
    \frac{d m}{d t} &= 0\\
    \frac{d D}{d t} &= m \gamma  i
    \end{aligned}
    :label: Mode

This is a common trick when solving systems of ODEs.  While equivalent in principle to using the appropriate quadrature scheme, this becomes especially important and convenient when adaptive time-stepping algorithms are used to solve the ODEs (i.e. there is no fixed time grid). Note that when doing so, :math:`M(0) = \int_0^0 \gamma m(\tau) i(\tau) d \tau = 0` is the initial condition.

The system :eq:`seir_system` and the supplemental equations can be written in vector form :math:`x := [s, e, i, r, R₀, m, c, M]` with parameter tuple :math:`p := (\sigma, \gamma, B, \eta)`

.. math::
    \begin{aligned}
    \frac{d x}{d t} &= F(x,t;p)\\
        &:= \begin{bmatrix}
        - \gamma \, R_0 \, s \,  i
        \\
        \gamma \, R_0 \,  s \,  i  - \sigma e
        \\
        \sigma \, e  - \gamma i
        \\
        \gamma i
        \\
         \eta (B(t) - R_0)
        \\
        0
        \\
        \sigma e
        \\
        m \, \gamma \, i
        \end{bmatrix}
    \end{aligned}
    :label: dfcv

Note that if :math:`B(t)` is time-invariant, then :math:`F(x)` is time-invariant as well.

Shocks to Transmission Rates
------------------------------

The basic reproduction number, :math:`R_0(t)`, can depend on degrees of randomness in behavior and implementation.  For example,

* Misinformation on Facebook spreading non-uniformly
* Large political rallies, elections, or protests
* Deviations in the implementation and timing of lockdown policy within demographics, locations, or businesses within the system.
* Aggregate shocks in opening/closing industries

To implement this, we will add on a diffusion term to :eq:`Rode` with an instantaneous volatility of :math:`\zeta \sqrt{R}`.  The scaling by the :math:`\sqrt{R_0}` ensure that the process (used in finance as the `CIR <https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model>`__ model of interest rates) stays weakly positive.  The heuristic explanation is that the variance of the shocks converges to zero as R₀ goes to zero, enabling the upwards drift to dominate.  See `here <https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model#Distribution>`__ for a heuristic description of when the process is weakly and strictly positive.

The notation for this `SDE <https://en.wikipedia.org/wiki/Stochastic_differential_equation#Use_in_probability_and_mathematical_finance>`__ is then

.. math::
    \begin{aligned}
    d R_{0t} &= \eta (B_t - R_{0t}) dt + \zeta \sqrt{R_{0t}} dW_t\\
    \end{aligned}
    :label: Rsde

where :math:`W` is standard Brownian motion (i.e a `Weiner Process <https://en.wikipedia.org/wiki/Wiener_process>`__.  This equation is used in the `Cox-Ingersoll-Ross <https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model>`__ and `Heston <https://en.wikipedia.org/wiki/Heston_model>`__ models of interest rates and stochastic volatility.

Heuristically, if :math:`\zeta = 0`, divide this equation by :math:`dt` and it nests the original ODE in :eq:`Rode`.

The general form of the SDE with these sorts of continuous-shocks is an extension of our :eq:`dfcv` definition .

.. math::
    \begin{aligned}
    d x_t &= F(x_t,t;p)dt + G(x_t,t;p) dW
    \end{aligned}

Here, it is convenient to :math:`d W` with the same dimension as :math:`x` where we can use the matrix :math:`G(x,t;p)` to associate the shocks with the appropriate :math:`x`.

As the shock only effects :math:`dR`, which is the 5th equation, define the matrix as

.. math::
    \begin{aligned}
    diag(G) &:= \begin{bmatrix} 0 & 0 & 0 & 0 & \zeta \sqrt{R_0} & 0 & 0 & 0 \end{bmatrix}
    \end{aligned}

First create a new settings generator, and then define a `SDEProblem <https://docs.sciml.ai/stable/tutorials/sde_example/#Example-2:-Systems-of-SDEs-with-Diagonal-Noise-1>`__  with Diagonal Noise.

We solve the problem with the `SOSRI <https://docs.sciml.ai/stable/solvers/sde_solve/#Full-List-of-Methods-1>`__ algorithm (Adaptive strong order 1.5 methods for diagonal noise Ito and Stratonovich SDEs)

.. code-block:: julia

    p_sde_gen = @with_kw ( T = 550.0, γ = 1.0 / 18, σ = 1 / 5.2, η = 1.0 / 20,
                    R̄₀ = 1.6, B = (t, p) -> p.R̄₀, m_0 = 0.01, ζ = 0.03, N = 3.3E8)

    p =  p_sde_gen()
    i_0 = 25000 / p.N
    e_0 = 75000 / p.N
    s_0 = 1.0 - i_0 - e_0
    R̄₀_0 = 1.5 * p.R̄₀  # TODO:  Do we want this, or start at 1.5?
    x_0 = [s_0, e_0, i_0, 0.0, R̄₀_0, p.m_0, 0.0, 0.0] # start in lockdown

    G(x, p, t) = [0, 0, 0, 0, p.ζ*sqrt(x[5]), 0, 0, 0]

    prob = SDEProblem(F, G, x_0, (0, p.T), p)
    sol_1 = solve(prob, SOSRI());
    @show length(sol_1.t);

With stochastic differential equations, a "solution" is akin to a simulation for a particular realization of the noise process. If we take two solutions and plot the number of infections, we will see differences over time:

.. code-block:: julia

    sol_2 = solve(prob, SOSRI())
    plot(sol_1, vars=[3], title = "Number of Infections", label = "Trajectory 1", lm = 2, xlabel = "t", ylabel = "i(t)")
    plot!(sol_2, vars=[3], label = "Trajectory 2", lm = 2, ylabel = "i(t)")


The same holds for other variables such as the flow of deaths:

.. code-block:: julia

    plot(sol_1, vars=[5], title = "Flow of Deaths", label = "Trajectory 1", lw = 2, xlabel = "t", ylabel = "c(t)")
    plot!(sol_2, vars=[5], label = "Trajectory 2", lw = 2)


While individual simulations are useful, you often want to look at an ensemble of trajectories of the SDE in order to get an accurate picture of how the system evolves on average. We can use the ``EnsembleProblem`` in order to have the solution compute multiple trajectories at once. The returned ``EnsembleSolution`` acts like an array of solutions but is imbued to plot recipes to showcase aggregate quantities. For example:

.. code-block:: julia

    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob, SOSRI(), EnsembleSerial(),trajectories = 10)
    plot(sol, vars = [3], title = "Infection Simulations", ylabel = "i(t)", xlabel = "t", lm = 2)


Or, more frequently, you may want to run many trajectories and plot quantiles, which can be automatically run in `parallel <https://docs.sciml.ai/stable/features/ensemble/>`_ using multiple threads, processes, or GPUs. Here we showcase ``EnsembleSummary`` which calculates summary information from an ensemble and plots the solution with the quantiles:

.. code-block:: julia

    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), trajectories = 1000)
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.95 quantiles
    plot(summ, idxs = (3,), title = "Quantiles of Infections Ensemble", ylabel = "i(t)", xlabel = "t", labels = "Middle 95% Quantile", legend = :topright)


While ensembles, you may want to perform transformations, such as calculating our daily deaths.  This can be done with an ``output_func`` executed with every simulation.

.. code-block:: julia

    function save_mortality(sol, ind)
        # Save x[5]=flow and N*x[8]=total from solution
        flow_and_total = [[x[5],p.N .* x[8]] for x in sol]
        return (DiffEqArray(flow_and_total,sol.t), false)
    end
    ensembleprob = EnsembleProblem(prob,  output_func = save_mortality)
    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), saveat = 0.5, trajectories = 1000)


Note that by using ``output_func`` we only end up storing the portions of the solution which we need. For large-scale stochastic simulations, this can be very helpful in reducing the amount of data stored for each simulation.

.. code-block:: julia

    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), saveat = 0.5, trajectories = 1000)
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.5, and 0.95 quantiles
    summ2 = EnsembleSummary(sol, quantiles = [0.25, 0.75])
    plot(summ, idxs = [5], title = "Daily Deaths (TBD)")
    plot!(summ2, idxs = [5], labels = "Middle 50%")

.. code-block:: julia

    plot(summ, idxs = [8], title = "Cumulative Death Percentage (TBD)")
    plot!(summ2, idxs = [8], labels = "Middle 50%")


Performance of these tends to be high, for example, rerunning out 1000 trajectories is measured in seconds on most computers with multithreading enabled.

.. code-block:: julia

    function F_static(x, p, t)

        s, e, i, r, R₀, m, c, D = x
        @unpack σ, γ, B, η = p

        return SA[-γ*R₀*s*i;     # ds/dt
                γ*R₀*s*i -  σ*e; # de/dt
                σ*e - γ*i;       # di/dt
                γ*i;             # dr/dt
                η*(B(t, p) - R₀);# dR₀/dt
                0.0;             # dm/dt
                σ*e;             # dc/dt
                m*γ*i;           # dd/dt
                ]
    end
    G_static(x, p, t) = SA[0, 0, 0, 0, p.ζ*sqrt(x[5]), 0, 0, 0]

    x_0_static = SVector{8}(x_0)
    prob_static = SDEProblem(F_static, G_static, x_0_static, (0, p.T), p)
    ensembleprob = EnsembleProblem(prob_static)
    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(),trajectories = 1000)
    @time solve(ensembleprob, SOSRI(), EnsembleThreads(),trajectories = 1000)

Note that these routines can also be auto-GPU accelerated by using
``EnsembleGPUArray()`` from `DiffEqGPU <https://github.com/SciML/DiffEqGPU.jl/>`

Technological Progress and Policy Tradeoffs
==============================================

While the randomness inherent in the :math:`R_0(t)` can explain some of the sources of uncertainty that come from behavioral shocks, we have been ignoring two other considerations.

First, technology, both in treatment and vaccination, is evolving and in an inherently non-deterministic way.  We will consider that the mortality rate :math:`m(t)` may evolve over time, as well as considering how a probability of vaccination development adds a path to the Removed state that does not require infection.


In order to add one more degree of realism to the tradeoffs, we will consider that the actual death rate is driven by the mortality :math:`m(t)` but also capacity constraints in the economy with respect to medical resources for the infected state.  In particular, we assume that if :math:`i(t) > \bar{i}`, then the available medical resources are exhausted, leading to quadratically increased death rate.

Second, the only social objective measure we can explore with the current framework is minimizing the total deaths.  That ignores the possible policy tradeoffs between minimizing deaths and the impact on the general economy.

While a particular planner may decide that the only relevant welfare criteria is aggregate mortality, that leads to implausibly extreme policy (e.g. set :math:`B(t) = 0` forever).  Hence, we will add a proxy for economic impact of COVID and the shutdown policy, summarized by :math:`u(t)` for excess COVID-related "unemployment".  A particular planner can then decide the weighting of the tradeoffs.

The policy choice :math:`B(t)` is then made a Markov process rather than current exogenous and deterministic one.

The inherent discreteness of medical innovations and policy changes provides us an opportunity to explore the use of Poisson and jump diffusion.

Mortality
---------

Imperfect mixing of different demographic groups could lead to aggregate shocks in mortality (e.g. if a retirement home is afflicted vs. an elementary school).  These sorts of relatively small changes might be best models as a continuous path process, so we will add a diffusion term with volatility :math:`\xi\sqrt{m}` to the :math:`m` process

In addition, there may be a variety of smaller medical interventions that are short of a vaccine, but still effect the :math:`m(t)` path.  For simplicity, we will assume that each innovation cuts the inherent mortality rate in half and arrives with intensity :math:`\alpha \geq 0`.  Combining these two leads to a jump diffusion process

.. math::
   \begin{aligned}
    d m_t & = \xi \sqrt{m_t} d W_t -  \frac{m}{2} d N_t^{\alpha}\\
    \end{aligned}
    :label: dmt

In that notation, the :math:`d W_t` is still the increment of the Brownian motion process while the  :math:`d N_t^{\alpha}` is a Poisson `counting process <https://en.wikipedia.org/wiki/Counting_process#:~:text=A%20counting%20process%20is%20a,(t)%20is%20an%20integer>`__ with rate :math:`\alpha`.


In previous versions of model, :eq:`Mode` had deaths as the integration of the death probability :math:`m` of the flow leaving the :math:`I` state.  i.e.  :math:`\frac{d D}{d t} = m \gamma  i`.  Instead, we will add an additional term that the death probability is :math:`\pi(m, i) := m + \psi \max(0, i - \bar{i})` increasing as :math:`i < \bar{i}`.  The ``min`` is used to ensure that the mortality rate never goes above 1.  The differential equation for cumulative deaths is then

.. math::
    \begin{aligned}
    \pi(m, i) &:= m + \psi \max(0, i - \bar{i})\\
    d D_t &= \pi(m_t, i_t)\gamma i_t dt\\
    \end{aligned}
    :label: Mode_nl

Vaccination
-------------

The develeopmment of a vaccine is a classic discrete event.  Define the availability of a vaccine as :math:`V_t` where the initial condition is :math:`V(0) = 0`.  We assume that with an arrival rate :math:`\theta` a vaccine will be developed, leading to the a jump to :math:`V = 1`.

The process for this leads to the following Jump process

.. math::
    \begin{aligned}
    d V_t & = (1 - V_t) d N_t^{\theta}
    \end{aligned}
    :label: dV

where the increment :math:`(1 - V_t)` is 1 if the vaccine hasn't been developed, and zero if it has.  While the constant arrival rate :math:`\theta` is convenient for a simple model, there is no reason it couldn't be time or state dependent in the code or model.

With a vaccine, the model ceases to be a simple SEIR model since it is possible move from :math:`S \to R` without passing through :math:`E` and :math:`I`.

As vaccines are not instantaneously delivered to all, we can let :math:`\nu` be the rate at which the vaccine is administered if available.  This leads to the following changes to :eq:`seir_system`

.. math::
   \begin{aligned}
        d s_t  & = \left(- \gamma \, R_t \, s_t \,  i _t - \nu \, V_t \, s_t\right)dt
        \\
         d r_t & =  (\gamma  i_t + \nu \, V_t \, s_t) dt
   \end{aligned}
   :label: seir_system_vacc


Unemployment and Lockdown Policy
----------------------------------

As a simple summary of the economic and social distortions due to the COVID, consider that an aggregate state :math:`u(t)` (indicating "excess" unemployment due to COVID) increases as the :math:`B(t)`policy deviates from the natural :math:`\bar{R}_0` level, but can also increase due to flow of deaths from COVID.

The second part of this is an important consideration: if the death rate is extremely high, then opening the economy may not help much as individuals are reluctant to return to normal economic activities.  We will assume that weights on these are :math:`\mu_B` and :math:`mu_M` respectively.

To represent the slow process of coming back to normal, we assume that the stock value :math:`u(t)` depreciates at rate :math:`\delta`.  Put together gives the ODE,

.. math::
   \begin{aligned}
        d u_t  & = (\mu_B(B_t - \bar{R}_0)^2 + \mu_d \pi(s_t,i_t)  -\delta) u_t dt\\
    \end{aligned}
   :label: du

The policy choice :math:`B(t)` is Markov, and will not consider implementation of forward looking behavior in this lecture.  For a simple example, consider a policy driven by myopic political incentives and driven entirely by the death rates.  If :math:`\pi_t > \bar{\pi}` then set it to the lockdwon level, :math:`B(t) = \bar{R}_{0, L}`.  Otherwise, leave it at the natural :math:`B(t) = \bar{R}_0`.

Without getting into the measure theory, this sort of bang-bang policy doesn't work as the process ceases to be a Levy Process (and much of the intuitive of considering measurability of stochastic processes and expected present discounted value fail).

To avoid these concerns, a standard trick is to make this a Levy Process by having a Poisson "policy evaluation" rate (which can be put arbitrarily high) where the policy is adjusted according to the rule.  Let that nuisance parameter by :math:`\Xi`, which gives the following pure-jump process for :math:`B(t)`

.. math::
   \begin{aligned}
        d B_t  & = J(m, i, B) dN^{\Xi}_t\\
        J(m, i, B) &:= -(\pi(i,m) > \bar{\pi})\max(B - \bar{R}_{0, L}, 0) + (\pi(i,m) \leq \bar{\pi})\max(B - \bar{R}_0, 0) \\
    \end{aligned}


Implementing the Complete System
---------------------------------

To our model, we have added in three new variables (:math:`u, V,` and :math:`B`) and a number of new parameters :math:`\bar{i}, \xi, \alpha, \theta, \nu, \mu_B, \mu_d, \bar{R}_{0, L}, \Xi, \delta, \psi`


Stacking the :math:`u, V, B` at the end of the existing :math:`x` vector, we add or modify using  :eq:`seir_system_vacc`, :eq:`Mode_nl`, and :eq:`du`.

Here, we will move from an "out of place" to an in-place differential equation.

.. code-block:: julia

    function F!(dx, x, p, t)

        s, e, i, r, R₀, m, c, D, u, V, B = x
        @unpack σ, γ, η, ī, ξ, α, θ, ν, μ_B, μ_d, R̄₀_L, Ξ, δ, π_bar, ψ, R̄₀, δ = p

        π = min(1.0, m + ψ * max(0.0, i > ī))

        dx[1] = -γ*R₀*s*i - ν*V*s;              # ds/dt
        dx[2] = γ*R₀*s*i -  σ*e;                # de/dt
        dx[3] = σ*e - γ*i;                      # di/dt
        dx[4] = γ*i + ν*V*s;                    # dr/dt
        dx[8] = π*γ*i;                          # dd/dt
        dx[7] = σ*e;                            # dc/dt
        dx[5] = η*(B - R₀);                     # dR₀/dt
        dx[6] = 0.0;                            # dm/dt
        dx[9] = (μ_B*(B - R̄₀)^2 + μ_d*π  - δ)*u;# du/dt
        dx[10] = 0.0;                           # dV/dt
        dx[11] = 0.0;                           # dB/dt
    end

Note that the ``V, B`` terms do not have a drift as they are pure jump processes.

Next, we need to consider how the variance term of the diffusion changes.  With the exception of the new Brownian
motion associated with the jump diffusion in :eq:`dmt`, everything else remains unchanged or zero.

.. code-block:: julia

    function G!(dx, x, p, t)
        @unpack ξ, ζ = p
        dx .= 0.0
        R₀ = x[5]
        m = x[6]
        dx[5] = R₀ <= 0.0 ? 0.0 : ζ*sqrt(R₀)
        dx[6] = m <= 0.0 ? 0.0 : ζ*sqrt(m)
    end

Setting the drift to be 0 if the :math:`R₀ \leq 0` or :math:`m \leq 0` is to deal with numerical instabilities of taking the square-root close to 0 (i.e. if :math:`m \approx 0`, it be slightly negative and taking the square root would be imaginary) 


Finally, we need to add in 3 jump processes which modify ``V, B, m`` separately.  The connection between a jump and a variable is not necessary, however.


Implementing the vaccination process,


.. code-block:: julia

    rate_V(x, p, t) = p.θ  # could be a function of time or state
    function affect_V!(integrator)
        integrator.u[10] = 1.0 # set the vacination state = 1
    end
    jump_V = ConstantRateJump(rate_V,affect_V!)


If the solver simulates an arrival rate of the jump, it calls the ``affect!`` function which takes in the current values through the ``integrator`` object and modifies the values directly.

The other two jump processes are,

.. code-block:: julia

    rate_B(x, p, t) = p.Ξ  # constant

    function affect_B!(integrator)
        @unpack σ, γ, η, ī, ξ, α, θ, ν, μ_B, μ_d, R̄₀_L, Ξ, δ, π_bar, ψ, R̄₀, δ = integrator.p

        m = integrator.u[6]
        i = integrator.u[3]
        π = min(1.0, m + ψ * max(0.0, i > ī))

        if π > π_bar
            integrator.u[11] = R̄₀_L
        else
            integrator.u[11] = R̄₀
        end
    end
    jump_B = ConstantRateJump(rate_B, affect_B!)

    rate_m(x, p, t) = p.α
    function affect_m!(integrator)
        integrator.u[6] = integrator.u[6]/2  # half the inherent mortatilty
    end
    jump_m = ConstantRateJump(rate_m, affect_m!)


Collecting the new parameters and providing an extension of the initial condition with no vaccine or accumulated :math:`u(t)`

.. code-block:: julia

    p_full_gen = @with_kw (
                      T = 550.0,
                      γ = 1.0 / 18,
                      σ = 1 / 5.2,
                      η = 1.0 / 20,
                      R̄₀ = 1.6,  # natural R_0
                      R̄₀_L = 0.5, # lockdown R_0 
                      m_0 = 0.01,
                      ζ = 0.03,
                      N = 3.3E8,
                      ī = 0.05,
                      ξ = 0.05,
                      α = 1/100,  # mean 100 days to innovation
                      θ = 1/300,  # mean 300 days to vaccine
                      ν = 0.05,
                      μ_B = 0.05,
                      μ_d = 0.1,
                      Ξ = 5.0,
                      δ = 0.07,
                      π_bar = 0.05,
                      ψ = 2.0
                    )

    p =  p_full_gen()
    i_0 = 25000 / p.N
    e_0 = 75000 / p.N
    s_0 = 1.0 - i_0 - e_0
    x_0 = [s_0, e_0, i_0, 0.0, p.R̄₀, p.m_0, 0.0, 0.0, 0.0, 0.0, p.R̄₀]

    prob = SDEProblem(F!, G!, x_0, (0, p.T), p)
    jump_prob = JumpProblem(prob, Direct(), jump_V, jump_B, jump_m)


Solving and simulating,

.. code-block:: julia

    sol = solve(jump_prob, SOSRI())
    plot(sol, vars = [8, 10])

TODO:  Chris, lets make sure this is right, pick some parameters.
TODO:  Lets show the same SEIR decomposition we had before as a proportion, but with a vaccination arrival.
TODO:  Maybe show a slice at time T = 550 or whatever of the distribution D(t) and u(t) as a histogram with 5, 50, and 95 percent confidence intervals?
TODO:  Then show how those two change as the myopic B(t) policy changes?
TODO:  Show how D(t) and u(t) change as the eta changes for a given policy?
