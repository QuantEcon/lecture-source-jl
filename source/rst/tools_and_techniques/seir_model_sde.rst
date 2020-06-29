.. include:: /_static/includes/header.raw

.. highlight:: julia

******************************************************************
:index:`Modeling COVID 19 with (Stochastic) Differential Equations`
******************************************************************

.. contents:: :depth: 2

Overview
=============

Coauthored with Chris Rackauckas


This is a Julia version of code for analyzing the COVID-19 pandemic.

The purpose of these notes is to introduce economists to quantitative modeling of infectious disease dynamics, and to modeling with ordinary and stochastic differential
equations.

The main objective is to study the impact of suppression through social distancing on the spread of the infection.

The focus is on US outcomes but the parameters can be adjusted to study
other countries.

In the first part, dynamics are modeled using a standard SEIR (Susceptible-Exposed-Infected-Removed) model
of disease spread, represented as a system of ordinary differential
equations where the number of agents is large and there are no exogenous stochastic shocks.

The first part of the model follows the notes from 
provided by `Andrew Atkeson <https://sites.google.com/site/andyatkeson/>`__

* `NBER Working Paper No. 26867 <https://www.nber.org/papers/w26867>`__ 
* `COVID-19 Working papers and code <https://sites.google.com/site/andyatkeson/home?authuser=0>`__

See further variations on the classic SIR model in Julia  `here <https://github.com/epirecipes/sir-julia>`__. 


We then look at extending the model to include policy-relevant aggregate shocks, and
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
    using Parameters, StaticArrays, Plots 


The SEIR Model
==============

In the version of the SEIR model we will analyze there are four states.

All individuals in the population are assumed to be in one of these four states.

The states are: susceptible (S), exposed (E), infected (I) and removed (R).

Comments:

* Those in state R have been infected and either recovered or died.

* Those who have recovered are assumed to have acquired immunity.

* Those in the exposed group are not yet infectious.



The interest is primarily in 

* the number of infections at a given time (which determines whether or not the health care system is overwhelmed) and
* how long the caseload can be deferred (hopefully until a vaccine arrives)
* how to model aggregate shocks related to policy, behavior, and medical innovations



Changes in the Infected State
-------------------------------

Within the SEIR model, the flow across states follows the path :math:`S \to E \to I \to R`.  Extensions of this model, such as SEIRS, relax lifetime immunity and allow
transitions from :math:`R \to S`.

The transitions between those states are governed by the following rates

* :math:`\beta(t)` is called the *transmission rate* (the rate at which individuals bump into others and expose them to the virus).
* :math:`\sigma` is called the *infection rate* (the rate at which those who are exposed become infected)
* :math:`\gamma` is called the *recovery rate* (the rate at which infected people recover or die).


In addition, the transmission rate can be rep-parameterized as: :math:`R(t) := \beta(t) / \gamma` where :math:`R(t)` has the interpretation as the *effective reproduction number* at time :math:`t`.  For this reason, we will work with the :math:`R(t)` reparameterization.

The notation is standard in the epidemiology literature - though slightly confusing, since :math:`R(t)` is different to
:math:`R`, the symbol that represents the removed state. Throughout the rest of the lecture, we will always use :math:`R` to represent the effective reproduction number, unless stated otherwise.

Assume that there is a constant population of size :math:`N` throughout, then define the proportion of people in each state as :math:`s := S/N` etc.  With this, and assuming a continuum approximation, the SEIR model can be written as 

.. math::
   \begin{aligned} 
        \frac{d s}{d t}  & = - \gamma \, R \, s \,  i  
        \\
        \frac{d e}{d t}   & = \gamma \, R \, s \,  i  - \sigma e 
        \\
         \frac{d i}{d t}  & = \sigma  e  - \gamma i
        \\
         \frac{d r}{d t}  & = \gamma  i         
   \end{aligned} 
   :label: seir_system

Here, :math:`dy/dt` represents the time derivative for the particular variable.

Since the states form a partition, we could reconstruct the "removed" fraction of the population as :math:`r = 1 - s - e - i`.  However, for further experiments and plotting it is harmless to keep it in the system.


Since we are using a continuum approximation, all individuals in the population are eventually infected when 
the transmission rate is positive and :math:`i(0) > 0`. 

We can begin writing the minimal code to solve the dynamics from a particular ``x_0 = [s_0, e_0, i_0, r_0]`` initial condition and parameter values.

First, define the system of equations

.. code-block:: julia

    function F_simple(x, p, t; γ = 1/18, R = 3.0, σ = 1/5.2)
        s, e, i, r = x

        return [-γ*R*s*i;       # ds/dt = -γRsi
                 γ*R*s*i -  σ*e;# de/dt =  γRsi -σe
                 σ*e - γ*i;     # di/dt =        σe -γi
                      γ*i;      # dr/dt =            γi
                ]          
    end

Written this way, we see that the four equations represent the one-directional transition from the susceptible to removed state, where the negative terms are outflows, and the positive ones inflows.  The equation associated with the state :math:`R` has inflows and no outflows, so any agents which end up in that state are "trapped".

Consequently, with no flow leaving the :math:`dr/dt` and strictly positive parameter values the ``R`` state is absorbing, and :math:`\lim_{t\to \infty} r(t) = 1`.  Crucial to this result is that individuals are perfectly divisible, and any arbitrarily small :math:`i > 0` leads to a strictly positive flow into the exposed state.

We will discuss this topic (and related topics of irreducibility) in the lecture on continuous-time Markov chains, as well as the limitations of these approximations when the discretness becomes essential (e.g. continuum approximations are incapable of modeling extinguishing of an outbreak).

Given this system, we choose an initial condition and a timespan, and create a ``ODEProblem`` encapsulating the system.

.. code-block:: julia

    i_0 = 1E-7
    e_0 = 4.0 * i_0
    s_0 = 1.0 - i_0 - e_0
    r_0 = 0.0
    x_0 = [s_0, e_0, i_0, r_0]  # initial condition

    tspan = (0.0, 350.0)  # ≈ 350 days
    prob = ODEProblem(F_simple, x_0, tspan)

With this, we can choose an ODE algorithm (e.g. a good default for non-stiff ODEs of this sort might be ``Tsit5()``, which is the Tsitouras 5/4 Runge-Kutta method).

.. code-block:: julia

    sol = solve(prob, Tsit5())
    plot(sol, labels = ["s" "e" "i" "r"], title = "SEIR Dynamics", lw = 2)


We did not provide either a set of time steps or a ``dt`` time step size to the ``solve``.  Most accurate and high-performance ODE solvers appropriate for this problem use adaptive time-stepping, changing the step size based the degree of curvature in the derivatives.


Or, as an alternative visualization, the proportions in each state over time

.. code-block:: julia

   areaplot(sol.t, sol', labels = ["s" "e" "i" "r"], title = "SEIR Proportions")


While implementing the system of ODEs in :math:`(s, e, i)`, we will extend the basic model to enable some policy experiments and calculations of aggregate values.

Extending the Model
-----------------------

First, we can consider some additional calculations such as the cumulative caseload (i.e., all those who have or have had the infection) as :math:`c = i + r`.  Differentiating that expression and substituting from the time-derivatives of :math:`i(t), r(t)` yields :math:`\frac{d c}{d t} = \sigma e`

We will assume that the transmission rate follows a process with a reversion to a value :math:`B(t)` which could conceivably be influenced by policy.  The intuition is that even if the targeted :math:`B(t)` was changed through social distancing/etc., lags in behavior and implementation would smooth out the transition, where :math:`\eta` governs the speed of :math:`R(t)` moves towards :math:`B(t)`. 

.. math::
    \begin{aligned} 
    \frac{d R}{d t} &= \eta (B - R)\\
    \end{aligned}         
    :label: Rode


Finally, let :math:`m(t)` be the mortality rate, which we will leave constant for now, i.e. :math:`\frac{d m}{d t} = 0`.  The cumulative deaths can be integrated through the flow :math:`\gamma i` entering the "Removed" state and define the cumulative number of deaths as :math:`M(t)`.

.. math::
    \begin{aligned}\\
    \frac{d m}{d t} &= 0\\
    \frac{d M}{d t} &= m \gamma  i
    \end{aligned}
    :label: Mode

While we could conceivably integrate the total deaths given the solution to the model, it is more convenient to use the integrator built into the ODE solver.  That is, we add :math:`d M(t)/dt` rather than calculating :math:`M(t) = \int_0^t \gamma m(\tau) i(\tau) d \tau` ex-post.

This is a common trick when solving systems of ODEs.  While equivalent in principle to using the appropriate quadrature scheme, this becomes especially important and convenient when adaptive time-stepping algorithms are used to solve the ODEs (i.e. there is no fixed time grid).

The system :eq:`seir_system` and the supplemental equations can be written in vector form :math:`x := [s, e, i, r, R, m, c, M]` with parameter tuple :math:`p := (\sigma, \gamma, B, \eta)`

.. math::
    \begin{aligned} 
    \frac{d x}{d t} &= F(x,t;p)\\
        &:= \begin{bmatrix}
        - \gamma \, R \, s \,  i  
        \\
        \gamma \, R \,  s \,  i  - \sigma e 
        \\
        \sigma \, e  - \gamma i
        \\
        \gamma i
        \\
         \eta (B(t) - R)
        \\
        0        
        \\
        \sigma e
        \\
        m \, \gamma \, i
        \end{bmatrix}
    \end{aligned}         
    :label: dfcv

Here note that if :math:`B(t)` is time-invariant, then :math:`F(x)` is time-invariant as well. 

Parameters
----------

Both :math:`\sigma` and :math:`\gamma` are thought of as fixed, biologically determined parameters.

As in Atkeson's note, we set

* :math:`\sigma = 1/5.2` to reflect an average incubation period of 5.2 days.
* :math:`\gamma = 1/18` to match an average illness duration of 18 days.
* :math:`B = R = 1.6` to match an **effective reproduction rate** of 1.6, and initially time-invariant
* :math:`m(t) = m_0 = 0.01` for a one-percent mortality rate


As we will initially consider the case where :math:`R(0) = B`, the value of :math:`\eta` will drop out of this first experiment.

Implementation
==============


First, construct our :math:`F` from :eq:`dfcv`

.. code-block:: julia

    function F(x, p, t)

        s, e, i, r, R, m, c, M = x
        @unpack σ, γ, B, η = p

        return [-γ*R*s*i;       # ds/dt
                γ*R*s*i -  σ*e; # de/dt
                σ*e - γ*i;      # di/dt
                γ*i;            # dr/dt
                η*(B(t, p) - R);# dR/dt
                0.0;            # dm/dt
                σ*e;            # dc/dt
                m*γ*i;          # dM/dt
                ]          
    end


Parameters
-------------

The baseline parameters are put into a named tuple generator (see previous lectures using ``Parameters.jl``) with default values discussed above.  

.. code-block:: julia

    p_gen = @with_kw ( T = 550.0, γ = 1.0 / 18, σ = 1 / 5.2, η = 1.0 / 20,
                      R̄ = 1.6, B = (t, p) -> p.R̄, m_0 = 0.01, N = 3.3E8)

Note that the default :math:`B(t)` function always equals :math:`\bar{R}`


Setting initial conditions, we will assume a fixed :math:`i, e`, :math:`r=0`, :math:`R(0) = \bar{B}`, and :math:`v(0) = 0.01` 

.. code-block:: julia

    p = p_gen()  # use all default parameters
 
    i_0 = 1E-7
    e_0 = 4.0 * i_0
    s_0 = 1.0 - i_0 - e_0

    x_0 = [s_0, e_0, i_0, 0.0, p.R̄, p.m_0, 0.0, 0.0]
    prob = ODEProblem(F, x_0, (0.0, p.T), p)


The ``tspan`` of ``(0.0, p.T)`` determines that the :math:`t` used by the solver.  The time scale needs to be consistent with the arrival
rate of the transition probabilities (i.e. the :math:`\gamma, \sigma` were chosen based on daily data).
The time period we investigate will be 550 days, or around 18 months:

Experiments
===========

Let's run some experiments using this code.

.. code-block:: julia

    sol = solve(prob, Tsit5())
    @show length(sol.t);

We see that the adaptive time-stepping used approximately 50 time-steps to solve this problem to the desires accuracy.  Evaluating the solver at points outside of those time-steps uses an interpolator consistent with the solution to the ODE.

See `here <https://docs.sciml.ai/stable/basics/solution/>`__ for details on analyzing the solution, and `here <https://docs.sciml.ai/stable/basics/plot/>`__ for plotting tools.  The built-in plots for the solutions provide all of the `attributes <https://docs.juliaplots.org/latest/tutorial/>`__ in `Plots.jl <https://github.com/JuliaPlots/Plots.jl>`__.

.. code_block:: julia

    # TODO: Chris, nice ways to resvale things or use two axis?
    plot(sol, vars = [7, 8], label = ["c(t)" "M(t)"], lw = 2, title = "Cumulative Infected and Total Mortality")


Experiment 1: Constant Reproduction Case
----------------------------------------

Let's start with the case where :math:`B(t) = R = \bar{R}` is constant.

We calculate the time path of infected people under different assumptions.

.. code-block:: julia

    R̄_vals = range(1.6, 3.0, length = 6)
    sols = [solve(ODEProblem(F, x_0, tspan, p_gen(R̄ = R̄)), Tsit5()) for R̄ in R̄_vals];
 
    # TODO: Probably clean ways to plot this , but don't know them!

    #R0_vals = np.linspace(1.6, 3.0, 6)
    #labels = [f'$R0 = {r:.2f}$' for r in R0_vals]
    #i_paths, c_paths = [], []
    #
    #for r in R0_vals:
    #    i_path, c_path = solve_path(r, t_vec)
    #    i_paths.append(i_path)
    #    c_paths.append(c_path)

    # Here's some code to plot the time paths.

    #def plot_paths(paths, labels, times=t_vec):
    #
    #    fig, ax = plt.subplots()
    #
    #    for path, label in zip(paths, labels):
    #        ax.plot(times, path, label=label)
    #        
    #    ax.legend(loc='upper left')
    #
    #    plt.show()

Let's plot current cases as a fraction of the population.

.. code-block:: julia

    #plot_paths(i_paths, labels)

As expected, lower effective transmission rates defer the peak of infections.

They also lead to a lower peak in current cases.

Here is cumulative cases, as a fraction of population:

.. code-block:: julia

    #plot_paths(c_paths, labels)


Experiment 2: Changing Mitigation
---------------------------------

Let's look at a scenario where mitigation (e.g., social distancing) is 
successively imposed, but the target (:math:`\bar{R}` is fixed)

To do this, we will have :math:`R(0) \neq \bar{R}` and examine the dynamics using the :math:`\frac{d R}{d t} = \eta (\bar{R} - R)` ODE.

.. Mathematica Verification
.. (\[Beta][t] /. 
..     First@DSolve[{\[Beta]'[t] == \[Eta] (b - \[Beta][t]), \[Beta][
..          0] == \[Beta]0}, \[Beta][t], 
..       t] ) == \[Beta]0 E^(-t \[Eta]) + (1 - 
..       E^(-t \[Eta])) b // FullSimplify

      
In the simple case, where :math:`B(t) = \bar{R}` is independent of the state, the solution to the ODE with :math:`R(0) = R_0` is :math:`R(t) = R_0 e^{-\eta t} + \bar{R}(1 - e^{-\eta t})`

We will examine the case where :math:`R(0) = 3` and then it falls to  to :math:`\bar{R} = 1.6` due to the progressive adoption of stricter mitigation measures.

The parameter ``η`` controls the rate, or the speed at which restrictions are
imposed.

We consider several different rates:

.. code-block:: julia

    #η_vals = 1/5, 1/10, 1/20, 1/50, 1/100
    #labels = [fr'$\eta = {η:.2f}$' for η in η_vals]

Let's calculate the time path of infected people, current cases, and mortality

.. code-block:: julia

    #i_paths, c_paths = [], []
    #
    #for η in η_vals:
    #    R0 = lambda t: R0_mitigating(t, η=η)
    #    i_path, c_path = solve_path(R0, t_vec)
    #    i_paths.append(i_path)
    #    c_paths.append(c_path)

    #plot_paths(i_paths, labels)

    #plot_paths(c_paths, labels)



Ending Lockdown
===============


The following is inspired by replicates `additional results <https://drive.google.com/file/d/1uS7n-7zq5gfSgrL3S0HByExmpq4Bn3oh/view>`__ by Andrew Atkeson on the timing of lifting lockdown.

Consider these two mitigation scenarios:

1. choose :math:`B(t)` to target :math:`\bar{R} = 0.5` for 30 days and then :math:`\bar{R} = 2` for the remaining 17 months. This corresponds to lifting lockdown in 30 days.

2. :math:`\bar{R} = 0.5` for 120 days and then :math:`\bar{R} = 2` for the remaining 14 months. This corresponds to lifting lockdown in 4 months.

For both of these, we will choose a large :math:`\eta` to focus on the case where rapid changes in the lockdown policy remain feasible.

The parameters considered here start the model with 25,000 active infections
and 75,000 agents already exposed to the virus and thus soon to be contagious.

.. code-block:: julia

    B_lift_early(t, p) = t < 30.0 ? 0.5 : 2.0
    B_lift_late(t, p) = t < 120.0 ? 0.5 : 2.0  
    p_early = p_gen(B = B_lift_early, η = 10.0)
    p_late = p_gen(B = B_lift_late, η = 10.0)

    
    # initial conditions 
    i_0 = 25000 / p_early.N
    e_0 = 75000 / p_early.N
    s_0 = 1.0 - i_0 - e_0
    R_0 = 0.5

    x_0 = [s_0, e_0, i_0, 0.0, R_0, p_early.m_0, 0.0, 0.0] # start in lockdown

    # create two problems, with rapid movement of R towards B(t)
    prob_early = ODEProblem(F, x_0, tspan, p_early)  
    prob_late = ODEProblem(F, x_0, tspan, p_late)


Unlike the previous examples, the :math:`B(t)` functions have discontinuities which might occur.  We can tell the adaptive time-stepping methods to ensure they include those points using ``tstops``

Let's calculate the paths:

.. code-block:: julia

    sol_early = solve(prob_early, Tsit5(), tstops = [30.0, 120.0])
    sol_late = solve(prob_late, Tsit5(), tstops = [30.0, 120.0])
    plot(sol_early, vars =[8], title = "Total Mortality", label = "Lift Early")
    plot!(sol_late, vars =[8], label = "Lift Late")

To calculate the daily death, calculate the :math:`\gamma i(t) m(t)`.

.. code-block:: julia

    flow_deaths(sol, p) = p.N * sol[3,:] .* sol[6,:] * p.γ

    daily_early = sol_early[3,:] .* sol_early[6,:] * p_early.γ
    daily_late = sol_late[3,:] .* sol_late[6,:] * p_late.γ

    plot(sol_early.t, flow_deaths(sol_early, p_early), title = "Flow Deaths", label = "Lift Early")
    plot!(sol_late.t, flow_deaths(sol_late, p_late), label = "Lift Late")    

Pushing the peak of curve further into the future may reduce cumulative deaths 
if a vaccine is found.


Despite its richness, the model above is fully deterministic.  The policy :math:`B(t)` could change over time, but only in predictable ways.

One source of randomness which would enter the model is considering the discretness of individuals.  This topic, the connection to between SDEs and the Langevin equations typically used in the approximation of chemical reactions in well-mixed media are explored in our lecture on continuous time Markov chains.

But rather than examining how granularity leads to aggregate fluctuations, we will concentrate on randomness that comes from aggregate changes in behavior or policy.

Introduction to SDEs: Aggregate Shocks to Transmission Rates
==============================================================

We will start by extending our model to include randomness in :math:`R(t)`, which makes it a system of Stochastic Differential Equations (SDEs).

Continuous-Time Stochastic Processes
-----------------------------------

In continuous-time, there is an important distinction between randomness that leads to continuous paths vs. those which may have jumps (which are almost surely right-continuous).  The most tractable of these includes the theory of `Levy Processes <https://en.wikipedia.org/wiki/L%C3%A9vy_process>`_.

**TBD:** Add definition of levy processes and the intuitive connection between stationary increments and independence of increments.  Also, is there intuition why additive processes in continuous time lead to additive SDEs/etc.?  Might be useful.

Among the appealing features of Levy Processes is that they fit well into the sorts of Markov modeling techniques that economists tend to use in discrete time...

Unlike in discrete-time, where a modeller has license to be creative, the rules of continuous-time stochastic processes are much more strict.  In practice, there are only two types of Levy Processes that can be used without careful measure theory.

#. `Weiner Processes <https://en.wikipedia.org/wiki/Wiener_process>`__ (as known as Brownian Motion) which leads to a diffusion equations, and is the only continuous-time Levy process with continuous paths
#. `Poisson Processes <https://en.wikipedia.org/wiki/Poisson_point_process>`__ with an arrival rate of jumps in the variable.

Every other Levy Process you will typically work with is composed of those building blocks (e.g. a `Diffusion Process <https://en.wikipedia.org/wiki/Diffusion_process>`__ such as Geometric Brownian Motion is a transformation of a Weiner process, and a `jump diffusion <https://en.wikipedia.org/wiki/Jump_diffusion#In_economics_and_finance>`__ is a diffusion process with a Poisson arrival of jumps).

In this section, we will look at example of a diffusion process.


Shocks to Transmission Rates
------------------------------

Consider that the effective transmission rate :math:`R(t)` could depend on degrees of randomness in behavior and implementation.  For example,

* Misinformation on Facebook spreading non-uniformly
* Large political rallies, elections, or protests
* Deviations in the implementation and timing of lockdown policy within demographics, locations, or businesses within the system.
* Aggregate shocks in opening/closing industries

To implement this, we will add on a diffusion term to :eq:`Rode` with an instantaneous volatility of :math:`\zeta \sqrt{R}`.  The scaling by the :math:`\sqrt{R}` ensures that the process can never go negative since the variance converges to zero as R goes to zero.

The notation for this `SDE <https://en.wikipedia.org/wiki/Stochastic_differential_equation#Use_in_probability_and_mathematical_finance>`__ is then

.. math::
    \begin{aligned} 
    d R_t &= \eta (B_t - R_t) dt + \zeta \sqrt{R_t} dW_t\\
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
    diag(G) &:= \begin{bmatrix} 0 & 0 & 0 & 0 & \zeta \sqrt{R} & 0 & 0 & 0 \end{bmatrix}
    \end{aligned}


Since these are additive shocks, we will not need to modify the :math:`F` from our equation.

First create a new settings generator, and then define a `SDEProblem <https://docs.sciml.ai/stable/tutorials/sde_example/#Example-2:-Systems-of-SDEs-with-Diagonal-Noise-1>`__  with Diagonal Noise. 

We solve the problem with the `SRA <https://docs.sciml.ai/stable/solvers/sde_solve/#Full-List-of-Methods-1>`__ algorithm (Adaptive strong order 1.5 methods for additive Ito and Stratonovich SDEs)

.. code-block:: julia

    p_sde_gen = @with_kw ( T = 550.0, γ = 1.0 / 18, σ = 1 / 5.2, η = 1.0 / 20,
                    R̄ = 1.6, B = (t, p) -> p.R̄, m_0 = 0.01, ζ = 0.03, N = 3.3E8)

    p =  p_sde_gen()
    i_0 = 25000 / p.N
    e_0 = 75000 / p.N
    s_0 = 1.0 - i_0 - e_0
    R_0 = 1.5 * p.R̄
    x_0 = [s_0, e_0, i_0, 0.0, R_0, p.m_0, 0.0, 0.0] # start in lockdown

    G(x, p, t) = [0, 0, 0, 0, p.ζ*sqrt(x[5]), 0, 0, 0]

    prob = SDEProblem(F, G, x_0, (0, p.T), p)
    sol_1 = solve(prob, SRA());
    @show length(sol_1.t);

With stochastic differential equations, a "solution" is akin to a simulation for a particular realization of the noise process.

Plotting the number of infections for these two realizations of the shock process

.. code-block:: julia

    # TODO: PICK BETTER PLOT
    plot(sol_1, vars = [2, 5] , title = "Stochastic R(t)", label = ["e(t)" "R(t)"])

If we solve this model a second time, and plot the flow of deaths, we can see differences over time

.. code-block:: julia

    sol_2 = solve(prob, SRA())
    plot(sol_1.t, flow_deaths(sol_1, p), title = "Daily Deaths", label = "sim 1")
    plot!(sol_2.t, flow_deaths(sol_2, p), label = "sim 2")



While individual simulations are useful, you often want to look at an ensemble of multiple trajectories of the SDE

.. code-block:: julia

    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob, SRA(), EnsembleSerial(),trajectories = 10)
    plot(sol, vars = [3], title = "Infection Simulations", label = "i(t)")


Or, more frequently, you may want to run many trajectories and plot quantiles, which can be automatically run in `parallel <https://docs.sciml.ai/stable/features/ensemble/>`_ using multiple threads, processes, or GPUs.

.. code-block:: julia    

    sol = solve(ensembleprob, SRA(), EnsembleThreads(),trajectories = 1000)
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.5, and 0.95 quantiles
    plot(summ, idxs = (3,), title = "Quantiles of Infections Ensemble", labels = "Middle 95% Quantile")


While ensembles, you may want to perform transformations, such as calculating our daily deaths.  This can be done with an ``output_func`` executed with every simulation.

.. code-block:: julia    

    # IS THRERE A BETTER WAY?
    function save_mortality(sol, ind)
        total = p.N * sol[8, :]
        flow = flow_deaths(sol, p)
        #TODO Add both total and flow
        return (DiffEqBase.build_solution(sol.prob, sol.alg, sol.t, total), false)
    end

    sol = solve(ensembleprob, SRA(), EnsembleThreads(), output_func = save_mortality, trajectories = 1000)

For large-scale stochastic simulations, we can also use the ``output_func`` to reduce the amount of data stored for each simulation.

.. code-block:: julia

    # CHRIS TODO: Can't Figure out plotting.  Maybe both of them in separate panes
    summ = EnsembleSummary(sol) # defaults to saving 0.05, 0.5, and 0.95 quantiles
    summ2 = EnsembleSummary(sol, quantiles = [0.25, 0.75])
    plot(summ, title = "Total and Daily Deaths (TBD)")
    plot!(summ2, labels = "Middle 50%")


Performance of these tends to be high, for example, rerunning out 1000 trajectories is measured in seconds on most computers with multithreading enabled.

.. 
.. CHRIS: THIS ENDED UP SLOWER and I am not sure why.  CAN ADD BACK IF WE CAN FIGURE OUT WHY
.. 
.. Furthermore, we can exploit Julia's generic programming to use a static array (or GPU, .. if available)
.. 
.. .. code-block:: julia
.. 
..     x_0_static = SVector{8}(x_0)
..     prob = SDEProblem(F, G, x_0_static, (0, p.T), p)
..     ensembleprob = EnsembleProblem(prob)
..     sol = solve(ensembleprob, SRA(), EnsembleThreads(),trajectories = 1000)
..     @time solve(ensembleprob, SRA(), EnsembleThreads(),trajectories = 1000);
..
.. TODO: link to the generic programming lecture, and explain why care was needed in the F definition.


Technological Progress and Policy Tradeoffs
==============================================

While the randomness inherent in the :math:`R(t)` can explain some of the sources of uncertainty that come from behavioral shocks, we have been ignoring two other considerations.

First, technology, both in treatment and vaccination, is evolving and in an inherently non-deterministic way.  We will consider that the mortality rate :math:`m(t)` may evolve over time, as well as considering how a probability of vaccination development adds a path to the Removed state that does not require infection.


In order to add one more degree of realism to the tradeoffs, we will consider that the actual death rate is driven by the mortality :math:`m(t)` but also capacity constraints in the economy with respect to medical resources for the infected state.  In particular, we assume that if :math:`i(t) > \bar{i}`, then the available medical resources are exhuasted, leading to quadratically increased death rate.

Second, the only social objective measure we can explore with the current framework is minimizing the total deaths.  That ignores the possible policy tradeoffs between minimizing deaths and the impact on the general economy.

While a particular planner may decide that the only relevant welfare criteria is aggregate mortality, that leads to implausibly extreme policy (e.g. set :math:`B(t) = 0` forever).  Hence, we will add a proxy for economic impact of COVID and the shutdown policy, summarized by :math:`u(t)` for excess COVID-related "unemployment".  A particular planner can then decide the weighting of the tradeoffs.

The policy choice :math:`B(t)` is then made a Markov process rather than current exogenous and deterministic one.

The inherent discretness of medical innovations and policy changes provides us an opportunity to explore the use of Poisson and jump diffusion. 

Mortality
---------

Imperfect mixing of different demographic groups could lead to aggregate shocks in mortality (e.g. if a retirement home is afflicted vs. an elementary school).  These sorts of relatively small changes might be best models as a continous path process, so we will add a diffusion term with volatility :math:`\xi\sqrt{m}` to the :math:`m` process

In addition, there may be a variety of smaller medical interventions that are short of a vaccine, but still effect the :math:`m(t)` path.  For simplicity, we will assume that each innovation cuts the inherent mortality rate in half and arrives with intensity :math:`\alpha \geq 0`.  Combining these two leads to a jump diffusion process

.. math::
   \begin{aligned} 
    d m_t & = \xi \sqrt{m_t} d W_t -  \frac{m}{2} d N_t^{\alpha}\\
    \end{aligned}
    :label: dmt

In that notation, the :math:`d W_t` is still the increment of the brownian motion process while the  :math:`d N_t^{\alpha}` is a Poisson `counting process <https://en.wikipedia.org/wiki/Counting_process#:~:text=A%20counting%20process%20is%20a,(t)%20is%20an%20integer>`__ with rate :math:`\alpha`.


In previous versions of model, :eq:`Mode` had deaths as the integration of the death probability :math:`m` of the flow leaving the :math:`I` state.  i.e.  :math:`\frac{d M}{d t} = m \gamma  i`.  Instead, we will add an additional term that the dealh probabilty is :math:`\pi(m, i) := m + \psi \max(0, i - \bar{i})` increasing as :math:`i < \bar{i}`.  The ``min`` is used to ensure that the mortality rate never goes above 1.  The differential equation for cumulative deaths is then

.. math::
    \begin{aligned}
    \pi(m, i) &:= m + \psi \max(0, i - \bar{i})\\
    d M_t &= \pi(m_t, i_t)\gamma i_t dt
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

As a simple summary of the economic and social distortions due to the COVID, consider that an aggregate state :math:`u(t)` (indicating "excess" unemployment due to COVID) increases as the :math:`B(t)`policy deviates from the natural :math:`\bar{R}` level, but can also increase due to flow of deaths from COVID.

The second part of this is an important consideration: if the death rate is extremely high, then opening the economy may not help much as individuals are reluctant to return to normal economic activities.  We will assume that weights on these are :math:`\mu_B` and :math:`mu_M` respectively.

To represent the slow process of coming back to normal, we assume that the stock value :math:`u(t)` depreciates at rate :math:`\delta`.  Put together gives the ODE,

.. math::
   \begin{aligned} 
        d u_t  & = (\mu_B(B_t - \bar{R})^2 + \mu_M \pi(s_t,i_t)  -\delta) u_t dt\\
    \end{aligned}
   :label: du

The policy choice :math:`B(t)` is Markov, and will not consider implementation of forward looking behavior in this lecture.  For a simple example, consider a policy driven by myopic political incentives and driven entirely by the death rates.  If :math:`\pi_t > \bar{\pi}` then set :math:`B(t) = R_0` and leave it at :math:`B(t) = \bar{R}` otherwise.

Without getting into the measure theory, this sort of bang-bang policy doesn't work as the process ceases to be a Levy Process (and much of the intuitive of considering measurability of stochastic processes and expected present discounted value fail).

To avoid these concerns, a standard trick is to make this a Levy Process by having a Poisson "policy evaluation" rate (which can be put arbitrarily high) where the policy is adjusted according to the rule.  Let that nuisance parameter by :math:`\Xi`, which gives the following pure-jump process for :math:`B(t)`

.. math::
   \begin{aligned} 
        d B_t  & = J(m, i, B) dN^{\Xi}_t\\
        J(m, i, B) &:= -(\pi(i,m) > \bar{\pi})\max(B - R_0, 0) + (\pi(i,m) \leq \bar{\pi})\max(B - \bar{R}, 0) \\
    \end{aligned}


Implementing the Complete System
---------------------------------

To our model, we have added in three new variables (:math:`u, V,` and :math:`B`) and a number of new parameters :math:`\bar{i}, \xi, \alpha, \theta, \nu, \mu_B, \mu_M, R_0, \Xi, \delta, \psi`


Stacking the :math:`u, V, B` at the end of the existing :math:`x` vector, we add or modify using  :eq:`seir_system_vacc`, :eq:`Mode_nl`, and :eq:`du`.

Here, we will move from an "out of place" to an in-place differential equation.

.. code-block:: julia

    function F!(dx, x, p, t)

        s, e, i, r, R, m, c, M, u, V, B = x
        @unpack σ, γ, η, ī, ξ, α, θ, ν, μ_B, μ_M, R_0, Ξ, δ, π_bar, ψ, R̄, δ = p

        π = min(1.0, m + ψ * max(0.0, i > ī))

        dx[1] = -γ*R*s*i - ν*V*s;              # ds/dt
        dx[2] = γ*R*s*i -  σ*e;                # de/dt
        dx[3] = σ*e - γ*i;                     # di/dt
        dx[4] = γ*i + ν*V*s;                   # dr/dt
        dx[5] = η*(B - R);                     # dR/dt
        dx[6] = 0.0;                           # dm/dt
        dx[7] = σ*e;                           # dc/dt
        dx[8] = π*γ*i;                         # dM/dt
        dx[9] = (μ_B*(B - R̄)^2 + μ_M*π  - δ)*u;# du/dt 
        dx[10] = 0.0;                          # dV/dt
        dx[11] = 0.0;                          # dB/dt                 
    end

Note that the ``V, B`` terms do not have a drift as they are pure jump processes.

Next, we need to consider how the variance term of the diffusion changes.  With the exception of the new Brownian
motion associated with the jump diffusion in :eq:`dmt`, everything else remains unchanged or zero.
    
.. code-block:: julia

    function G!(dx, x, p, t)
        @unpack ξ, ζ = p
        dx .= 0.0
        R = x[5]
        m = x[6]
        dx[5] = R <= 0.0 ? 0.0 : ζ*sqrt(R)
        dx[6] = m <= 0.0 ? 0.0 : ζ*sqrt(m)
    end


Finally, we need to add in 3 jump processes which modify ``V, B, m`` separately.  The connection between a jump and a variable is not necsesary, however.


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
        @unpack σ, γ, η, ī, ξ, α, θ, ν, μ_B, μ_M, R_0, Ξ, δ, π_bar, ψ, R̄, δ = integrator.p

        m = integrator.u[6]
        i = integrator.u[3]
        π = min(1.0, m + ψ * max(0.0, i > ī))

        if π > π_bar
            integrator.u[11] = R_0
        else
            integrator.u[11] = R̄
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

    p_full_gen = @with_kw ( T = 550.0, γ = 1.0 / 18, σ = 1 / 5.2, η = 1.0 / 20,
                      R̄ = 1.6, R_0 = 0.5, m_0 = 0.01, ζ = 0.03, N = 3.3E8,
                      ī = 0.05,
                      ξ = 0.05,
                      α = 1/100,  # mean 100 days to innovation
                      θ = 1/300,  # mean 200 days to vaccine
                      ν = 0.05,
                      μ_B = 0.05,
                      μ_M = 0.1,
                      Ξ = 5.0,
                      δ = 0.07,
                      π_bar = 0.05,
                      ψ = 2.0
                    )

    p =  p_full_gen()
    i_0 = 25000 / p.N
    e_0 = 75000 / p.N
    s_0 = 1.0 - i_0 - e_0
    x_0 = [s_0, e_0, i_0, 0.0, p.R̄, p.m_0, 0.0, 0.0, 0.0, 0.0, p.R̄]
  
    prob = SDEProblem(F!, G!, x_0, (0, p.T), p)
    jump_prob = JumpProblem(prob, Direct(), jump_V, jump_B, jump_m)


Solving and simulating,

.. code-block:: julia

    sol = solve(jump_prob, SRIW1())
    plot(sol, vars = [8, 10])

TODO:  Chris, lets make sure this is right, pick some parameters.
TODO:  Lets show the same SEIR decomposition we had before as a proportion, but with a vaccination arrival.
TODO:  Maybe show a slice at time T = 550 or whatever of the distribution M(t) and u(t) as a histogram with 5, 50, and 95 percent confidence intervals?
TODO:  Then show how those two change as the myopic B(t) policy changes?
TODO:  Show how M(t) and u(t) change as the eta changes for a given policy?

ROUGH NOTES ON THE SCIML LECTURE
=====================================

The hope is that we have all or almost all of the model elements developed here, so that we can
focus on the machine learning/etc. in the following lecture and not implement any new model features (i.e. just shut things down as required).


Some thoughts on the SciML lecture which would be separate:

1.  Bayesian estimation and model uncertainty.  Have uncertainty on the parameter such as the :math:`m_0` parameter.  Show uncertainty that comes after the estimation of the parameter.
2.  Explain how Bayesian priors are a form of regularization.  With rare events like this, you will always have
more parameters than data.
3.  Solve the time-0 optimal policy problem using Flux of choosing the :math:`B` policy in some form.  For the objective, consider the uncertainty and the asymmetry of being wrong.
4. If we wanted to add in a model element, I think that putting in an asymptomatic state would be very helpful for explaining the role of uncertainty in evaluating the counter-factuals.
5. Solve for the optimal Markovian B(t) policy through simulation and using Flux. 
