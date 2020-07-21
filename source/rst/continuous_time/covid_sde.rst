.. _covid_sde:

.. include:: /_static/includes/header.raw

.. highlight:: julia

****************************************************************************
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

Here we extend the model to include policy-relevant aggregate shocks.

Continuous-Time Stochastic Processes
------------------------------------

In continuous-time, there is an important distinction between randomness that leads to continuous paths vs. those which may have jumps (which are almost surely right-continuous).  The most tractable of these includes the theory of `Levy Processes <https://en.wikipedia.org/wiki/L%C3%A9vy_process>`_.

.. **TBD:** Add definition of levy processes and the intuitive connection between stationary increments and independence of increments.

Among the appealing features of Levy Processes is that they fit well into the sorts of Markov modeling techniques that economists tend to use in discrete time.

Unlike in discrete-time, where a modeller has license to be creative, the rules of continuous-time stochastic processes are much stricter.  In practice, there are only two types of Levy Processes that can be used without careful measure theory.

#. `Weiner Processes <https://en.wikipedia.org/wiki/Wiener_process>`__ (as known as Brownian Motion) which leads to a diffusion equations, and is the only continuous-time Levy process with continuous paths
#. `Poisson Processes <https://en.wikipedia.org/wiki/Poisson_point_process>`__ with an arrival rate of jumps in the variable.

Every other Levy Process can be represented by these building blocks (e.g. a `Diffusion Process <https://en.wikipedia.org/wiki/Diffusion_process>`__ such as Geometric Brownian Motion is a transformation of a Weiner process, and a `jump diffusion <https://en.wikipedia.org/wiki/Jump_diffusion#In_economics_and_finance>`__ is a diffusion process with a Poisson arrival of jumps).

In this section, we will examine shocks driven by Brownian motion, as the prototypical Stochastic Differential Equation (SDE).


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

    using OrdinaryDiffEq, StochasticDiffEq
    using Parameters, StaticArrays, Plots


The Basic SIR/SIRD Model
=========================

To demonstrate another common `compartmentalized model <https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#Elaborations_on_the_basic_SIR_model>`__ we will change the SEIR model to remove the exposed state, and more carefully manage the death state, D.

The states are: susceptible (S), infected (I), resistant (R), or dead (D).


Comments:

* Unlike the previous SEIR model, the R state is only for those recovered, alive, and currently resistant.  

* As before, we start by assuming those have recovered have acquired immunity.

* Later, we could consider transitions from R to S if resistance is not permanent due to virus mutation, etc.

Transition Rates
-------------------------------

See the :doc:`previous lecture <seir_model>`, for a more detailed development of the model.

* :math:`\beta(t)` is called the *transmission rate* or *effective contact rate* (the rate at which individuals bump into others and expose them to the virus)
* :math:`\gamma` is called the *resolution rate* (the rate at which infected people recover or die)
* :math:`\delta(t) \in [0, 1]` is the *death probability*
* As before, we re-parameterize as :math:`R_0(t) := \beta(t) / \gamma`, where :math:`R_0` has previous interpretation

Jumping directly to the equations in :math:`s, i, r, d` already normalized by :math:`N`,

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
   :label: SIRD


Introduction to SDEs: Aggregate Shocks to Transmission Rates
==============================================================

We start by extending our model to include randomness in :math:`R_0(t)`, which makes it a system of Stochastic Differential Equations (SDEs).

Shocks to Transmission Rates
------------------------------

As before, we assume that the basic reproduction number, :math:`R_0(t)`, follows a process with a reversion to a value :math:`\bar{R}_0(t)` which could conceivably be influenced by policy.  The intuition is that even if the targeted :math:`\bar{R}_0(t)` was changed through social distancing/etc., lags in behavior and implementation would smooth out the transition, where :math:`\eta` governs the speed of :math:`R_0(t)` moves towards :math:`\bar{R}_0(t)`.


Beyond changes in policy, :math:`R_0(t)` can depend on degrees of randomness in behavior and implementation.  For example,

* Misinformation on Facebook spreading non-uniformly
* Large political rallies, elections, or protests
* Deviations in the implementation and timing of lockdown policy within demographics, locations, or businesses within the system.
* Aggregate shocks in opening/closing industries

To implement this, we will add on a diffusion term to with an instantaneous volatility of :math:`\zeta \sqrt{R_0}`.

* This equation is used in the `Cox-Ingersoll-Ross <https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model>`__ and `Heston <https://en.wikipedia.org/wiki/Heston_model>`__ models of interest rates and stochastic volatility.
* The scaling by the :math:`\sqrt{R_0}` ensure that the process stays weakly positive.  The heuristic explanation is that the variance of the shocks converges to zero as R₀ goes to zero, enabling the upwards drift to dominate.
* See `here <https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model#Distribution>`__ for a heuristic description of when the process is weakly and strictly positive.

The notation for this `SDE <https://en.wikipedia.org/wiki/Stochastic_differential_equation#Use_in_probability_and_mathematical_finance>`__ is then

.. math::
    \begin{aligned}
    d R_{0t} &= \eta (\bar{R}_{0t} - R_{0t}) dt + \zeta \sqrt{R_{0t}} dW_t\\
    \end{aligned}
    :label: Rsde

where :math:`W` is standard Brownian motion (i.e a `Weiner Process <https://en.wikipedia.org/wiki/Wiener_process>`__.

Heuristically, if :math:`\zeta = 0`, divide this equation by :math:`dt` and it nests the original ODE used in the previous lecture

Mortality Rates
----------------

Unlike the previous lecture, we will build up towards mortality rates which change over time.

Imperfect mixing of different demographic groups could lead to aggregate shocks in mortality (e.g. if a retirement home is afflicted vs. an elementary school).  These sorts of relatively small changes might be best models as a continuous path process.

Let :math:`\delta(t)` be the mortality rate.

In addition,

* Assume that the base mortality rate is :math:`\bar{\delta}`, which acts as the mean of the process, reverting at rate :math:`\xi`.
* The diffusion term has a volatility :math:`\xi\sqrt{\delta (1 - \delta)}`.
* As the process gets closer to either :math:`\delta = 1` or :math:`\delta = 0`, the volatility goes to 0, which acts as a force to allow the mean reversion to keep the process within the bounds
* Unlike the well-studied Cox-Ingersoll-Ross model, we make no claims on the long-run behavior of this process, but will be examining the behavior on a small timescale so this is not an issue.

Given this, the stochastic process for the mortality rate is,
 
.. math::
    \begin{aligned}
    d \delta_t & = \theta (\bar{\delta} - \delta_t) dt + \xi \sqrt{(\delta_t (1-\delta_t)} d W_t\\
    \end{aligned}
    :label: dmt

Where the :math:`W_t` Brownian motion is independent from the previous process. 

System of SDEs
---------------
The system :eq:`SIRD` can be written in vector form :math:`x := [s, i, r, d, R₀, \delta]` with parameter tuple parameter tuple :math:`p := (\gamma, \eta, \sigma, \theta, \xi)`

The general form of the SDE is.

.. math::
    \begin{aligned}
    d x_t &= F(x_t,t;p)dt + G(x_t,t;p) dW
    \end{aligned}


With the drift,

.. math::
    \begin{aligned}
    F(x,t;p)\\
        &:= \begin{bmatrix}
        - \gamma \, R_0 \, s \,  i
        \\
        \gamma \, R_0 \,  s \,  i  - \gamma i
        \\
        (1-\delta) \gamma i
        \\
        \delta \gamma i
        \\
        \eta (\bar{R}_0(t) - R_0)
        \\
        \theta (\bar{\delta} - \delta)
        \\
        \end{bmatrix}
    \end{aligned}
    :label: dfcvsde


Here, it is convenient but not necessary for :math:`d W` to have the same dimension as :math:`x` where we can use the matrix :math:`G(x,t;p)` to associate the shocks with the appropriate :math:`x` (e.g. diagonal noise, or a covariance matrix).

As the two shock only effects :math:`d R_0` and :math:`d \delta` (i.e. the 5th and 6th equations) and are independent, define the covariance matrix as

.. math::
    \begin{aligned}
    diag(G(x, t)) &:= \begin{bmatrix} 0 & 0 & 0 & 0 & \sigma \sqrt{R_0} & \xi \sqrt{\delta (1-\delta)} \end{bmatrix}
    \end{aligned}
    :label: dG

.. 
.. Daily Deaths
.. --------------
.. 
.. Outside of the system of equations, a key calculation will be the :math:`d/dt D(t)`, i.e. the daily deaths, where our timescale is already in days.
.. 
.. 
.. Define :math:`\Delta D \approx d/dt D(t)` where we assume that the parameters are roughly fixed over a 1-day time-horizon.  In that case, we define :math:`\Delta D := N \delta \gamma i`.
.. 

Implementation
----------------

First, construct our :math:`F` from :eq:`dfcvsde` and :math:`G` from :eq:`dG`

.. code-block:: julia

    function F(x, p, t)
        s, i, r, d, R₀, δ = x
        @unpack γ, R̄₀, η, σ, ξ, θ, δ_bar = p

        return [-γ*R₀*s*i;        # ds/dt
                γ*R₀*s*i - γ*i;   # di/dt
                (1-δ)*γ*i;        # dr/dt
                δ*γ*i;            # dd/dt
                η*(R̄₀(t, p) - R₀);# dR₀/dt
                θ*(δ_bar - δ);    # dδ/dt
                ]
    end

    function G(x, p, t)
        s, i, r, d, R₀, δ = x
        @unpack γ, R̄₀, η, σ, ξ, θ, δ_bar = p

        return [0; 0; 0; 0; σ*sqrt(R₀); ξ*sqrt(δ * (1-δ))]
    end

Next create a settings generator, and then define a `SDEProblem <https://docs.sciml.ai/stable/tutorials/sde_example/#Example-2:-Systems-of-SDEs-with-Diagonal-Noise-1>`__  with Diagonal Noise.

.. code-block:: julia

    p_gen = @with_kw ( T = 550.0, γ = 1.0 / 18, η = 1.0 / 20,
                    R₀_n = 1.6, R̄₀ = (t, p) -> p.R₀_n, δ_bar = 0.01, σ = 0.02, ξ = 0.005, θ = 0.2, N = 3.3E8)
    p =  p_gen()  # use all defaults
    i_0 = 25000 / p.N
    r_0 = 0.0
    d_0 = 0.0
    s_0 = 1.0 - i_0 - r_0 - d_0
    R̄₀_0 = 0.5  # starting in lockdown
    δ_0 = p.δ_bar
    x_0 = [s_0, i_0, r_0, d_0, R̄₀_0, δ_0]

    prob = SDEProblem(F, G, x_0, (0, p.T), p)

We solve the problem with the `SOSRI <https://docs.sciml.ai/stable/solvers/sde_solve/#Full-List-of-Methods-1>`__ algorithm (Adaptive strong order 1.5 methods for diagonal noise Ito and Stratonovich SDEs).

.. code-block:: julia

    sol_1 = solve(prob, SOSRI());
    @show length(sol_1.t);

As in the deterministic case of the previous lecture, we are using an adaptive time-stepping method.  However, since this is a SDE, the number of timesteps will change with different shock realizations.

Note: The structure of :math:`G(x, t)` will determine the best algorithms.  For example, if :math:`G` is independent of the state, then the noise is additive and ``SRA1`` is an appropriate algorithm for non-stiff equations. 

With stochastic differential equations, a "solution" is akin to a simulation for a particular realization of the noise process. If we take two solutions and plot the number of infections, we will see differences over time:

.. code-block:: julia

    sol_2 = solve(prob, SOSRI())
    plot(sol_1, vars=[2], title = "Number of Infections", label = "Trajectory 1", lm = 2, xlabel = "t", ylabel = "i(t)")
    plot!(sol_2, vars=[2], label = "Trajectory 2", lm = 2, ylabel = "i(t)")

The same holds for other variables such as the cumulative deaths, mortality, and :math:`R_0`:

.. code-block:: julia

    plot_1 = plot(sol_1, vars=[4], title = "Cumulative Death Proportion", label = "Trajectory 1", lw = 2, xlabel = "t", ylabel = "d(t)", legend = :topleft)
    plot!(plot_1, sol_2, vars=[4], label = "Trajectory 2", lw = 2)
    plot_2 = plot(sol_1, vars=[3], title = "Cumulative Recovered Proportion", label = "Trajectory 1", lw = 2, xlabel = "t", ylabel = "d(t)", legend = :topleft)
    plot!(plot_2, sol_2, vars=[3], label = "Trajectory 2", lw = 2)
    plot_3 = plot(sol_1, vars=[5], title = "R_0 transition from lockdown", label = "Trajectory 1", lw = 2, xlabel = "t", ylabel = "R_0(t)")
    plot!(plot_3, sol_2, vars=[5], label = "Trajectory 2", lw = 2)
    plot_4 = plot(sol_1, vars=[6], title = "Mortality Rate", label = "Trajectory 1", lw = 2, xlabel = "t", ylabel = "delta(t)", ylim = (0.006, 0.014))
    plot!(plot_4, sol_2, vars=[6], label = "Trajectory 2", lw = 2)
    plot(plot_1, plot_2, plot_3, plot_4, size = (900, 600))


Ensembles
-----------

While individual simulations are useful, you often want to look at an ensemble of trajectories of the SDE in order to get an accurate picture of how the system evolves on average. We can use the ``EnsembleProblem`` in order to have the solution compute multiple trajectories at once. The returned ``EnsembleSolution`` acts like an array of solutions but is imbued to plot recipes to showcase aggregate quantities. For example:

.. code-block:: julia

    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob, SOSRI(), EnsembleSerial(),trajectories = 10)
    plot(sol, vars = [2], title = "Infection Simulations", ylabel = "i(t)", xlabel = "t", lm = 2)


Or, more frequently, you may want to run many trajectories and plot quantiles, which can be automatically run in `parallel <https://docs.sciml.ai/stable/features/ensemble/>`_ using multiple threads, processes, or GPUs. Here we showcase ``EnsembleSummary`` which calculates summary information from an ensemble and plots the solution with the quantiles:

.. code-block:: julia

    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), trajectories = 1000)
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.95 quantiles
    plot(summ, idxs = (2,), title = "Quantiles of Infections Ensemble", ylabel = "i(t)", xlabel = "t", labels = "Middle 95% Quantile", legend = :topright)

In addition, you can calculate more quantiles and stack graphs

.. code-block:: julia

    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), trajectories = 1000)
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.95 quantiles
    summ2 = EnsembleSummary(sol, quantiles = (0.25, 0.75))

    plot(summ, idxs = (2,4,5,6),
        title = ["Proportion Infected" "Proportion Dead" "R_0" "delta"],
        ylabel = ["i(t)" "d(t)" "R_0(t)" "delta(t)"], xlabel = "t",
        legend = [:topleft :topleft :bottomright :bottomright],
        labels = "Middle 95% Quantile", layout = (2, 2), size = (900, 600))
    plot!(summ2, idxs = (2,4,5,6),
        labels = "Middle 50% Quantile", legend =  [:topleft :topleft :bottomright :bottomright])


Some important additional features of the ensemble and SDE infrastructure are 

* Plotting https://diffeq.sciml.ai/stable/basics/plot/
* `Noise Processes <https://diffeq.sciml.ai/stable/features/noise_process/>`__ and `Non-diagonal noise <https://diffeq.sciml.ai/stable/tutorials/sde_example/#Example-4:-Systems-of-SDEs-with-Non-Diagonal-Noise-1>`__ : Variations on the simple diagonal noise example provided above

Lifting Lockdown
-----------------

Consider a variation on the previous policy where the lockdown is relaxed at a slower speed.

We will shut down the shocks to the mortality rate to focus on the variation caused by the volatility in :math:`R_0(t)`.

We can overlay the ensembles to see the impact on the proportion dead

.. code-block:: julia

    summ_1 = EnsembleSummary(solve(EnsembleProblem(SDEProblem(F, G, x_0, (0, 120.0), p_gen(η = 1/50, ξ = 0.0))), SOSRI(), EnsembleThreads(), trajectories = 1000))
    summ_2 = EnsembleSummary(solve(EnsembleProblem(SDEProblem(F, G, x_0, (0, 120.0), p_gen(η = 1/20, ξ = 0.0))), SOSRI(), EnsembleThreads(), trajectories = 1000))
    plot(summ_1, idxs = (4,5),
        title = ["Proportion Dead" "R_0"],
        ylabel = ["d(t)" "R_0(t)"], xlabel = "t",
        legend = [:topleft :bottomright],
        labels = "Middle 95% Quantile, eta = 1/50", layout = (2, 1), size = (900, 900), fillalpha = 0.5)
    plot!(summ_2, idxs = (4,5),
        legend = [:topleft :bottomright],
        labels = "Middle 95% Quantile, eta = 1/20", size = (900, 900), fillalpha = 0.5)

While the the mean of the :math:`d(t)` increases, almost mechanically, we see that the 95% quantile for later time periods is also much larger - even after the :math:`R_0` has converged.

That is, volatile contact rates (and hence :math:`R_0`) can catastrophic worst-case scenarios due to the dynamics of the system.

Additional Calculations
-------------------------

Furthermore, in the ensembles, you may want to perform calculations and reductions.

In our case, we need to fill in the placeholder for the daily deaths, :math:`\Delta D`.

This can be done with an ``output_func`` executed at the end of every simulation and before the data is collected

* See `
*  A benefit of using ``output_func`` is that we only end up storing the portions of the solution which we need. For large-scale stochastic simulations, this can be very helpful in reducing the amount of data stored for each simulation.
* Note: an additional transformation is the `reduction <https://diffeq.sciml.ai/stable/features/ensemble/#Example-3:-Using-the-Reduction-to-Halt-When-Estimator-is-Within-Tolerance-1>`__ which allows you the output of ensembles as they are completed in parallel.  If the ensemble uses an ``output_func`` then it is a reduction on the output of that function.


.. code-block:: julia
    
    function save_ensemble_data(sol, ind)        
        flow_and_total = [[x[5], p.N .* x[8]] for x in sol]
        return (DiffEqArray(flow_and_total, sol.t), false)
    end
    saveat = 1.0  # store data at daily frequency, not a dt step-size
    trajectories = 1000

    ensembleprob = EnsembleProblem(prob,  output_func = save_ensemble_data)
    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), saveat = saveat, trajectories = trajectories)


In addition, note the use of ``saveat`` in the ``solve`` function.

This option will only save data when running the ensembles at that frequency (or, we could give it a vector of dates instead).  Unless the saveat happened to be exactly at the timestep, it will use the built-in interpolation of the solver.  i.e. this has nothing to do with the step-size of the solver itself. 

Since we are using adaptive time-stepping methods here, we would otherwise not have the ``sol.t`` for different simulations coincide, so this is a necessary step when returning a ``DiffEqArray``.  If you are not using an ``output_func``, then the ensemble code will automatically use interpolation to make times comparable.


.. code-block:: julia

    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(), saveat = 1.0, trajectories = 1000)
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.5, and 0.95 quantiles
    summ2 = EnsembleSummary(sol, quantiles = [0.25, 0.75])
    plot(summ, idxs = [5], title = "Daily Deaths (TBD)")
    plot!(summ2, idxs = [5], labels = "Middle 50%")

.. code-block:: julia

    plot(summ, idxs = [4], labels = "Middle 95%", title = "Cumulative Death Proportion")
    plot!(summ2, idxs = [4], labels = "Middle 50%")


Static Arrays and GPUs
-------------------------


Performance of these tends to be high, for example, rerunning out 1000 trajectories is measured in seconds on most computers with multi-threading enabled.


In addition, we can write versions with static arrays given the small dimension of the system.

.. code-block:: julia

    function F_static(x, p, t)
        s, i, r, d, R₀, δ = x
        @unpack γ, R̄₀, η, σ, ξ, θ, δ_bar = p

        return SA[-γ*R₀*s*i;      # ds/dt
                γ*R₀*s*i - γ*i;   # di/dt
                (1-δ)*γ*i;        # dr/dt
                δ*γ*i;            # dd/dt
                η*(R̄₀(t, p) - R₀);# dR₀/dt
                θ*(δ_bar - δ);    # dδ/dt
                ]
    end

    function G_static(x, p, t)
        s, i, r, d, R₀, δ = x
        @unpack γ, R̄₀, η, σ, ξ, θ, δ_bar = p

        return SA[0; 0; 0; 0; σ*sqrt(R₀); ξ*sqrt(δ * (1-δ))]
    end

    x_0_static = SVector{6}(x_0)
    prob_static = SDEProblem(F_static, G_static, x_0_static, (0, p.T), p)
    ensembleprob = EnsembleProblem(prob_static)
    sol = solve(ensembleprob, SOSRI(), EnsembleThreads(),trajectories = 1000)
    @time solve(ensembleprob, SOSRI(), EnsembleThreads(),trajectories = 1000)

Note that these routines can also be auto-GPU accelerated by using
``EnsembleGPUArray()`` from `DiffEqGPU <https://github.com/SciML/DiffEqGPU.jl/>`