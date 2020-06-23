.. include:: /_static/includes/header.raw

.. highlight:: julia

******************************************************************
:index:`Modeling COVID 19 with (Stochastic) Differential Equations`
******************************************************************

.. contents:: :depth: 2



Overview
=============

This is a Julia version of the code for analyzing the COVID-19 pandemic.

The purpose of these notes is to introduce economists to quantitative modeling
of infectious disease dynamics, and to modeling with ordinary and stochastic differential
equations.

The main objective is to study the impact of suppression through social
distancing on the spread of the infection.

Dynamics are modeled using a standard SEIR (Susceptible-Exposed-Infected-Removed) model
of disease spread, represented as a system of ordinary differential
equations when the number of agents is large and there are no exogenous stochastic shocks.

The focus is on US outcomes but the parameters can be adjusted to study
other countries.


The first part of the model follows the notes from 
provided by `Andrew Atkeson <https://sites.google.com/site/andyatkeson/>`__

* `NBER Working Paper No. 26867 <https://www.nber.org/papers/w26867>`__ 
* `COVID-19 Working papers and code <https://sites.google.com/site/andyatkeson/home?authuser=0>`__

See further variations on the classic SIR model in Julia  `here <https://github.com/epirecipes/sir-julia>`__. 



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


The SEIR Model
=============

In the version of the SEIR model we will analyze there are four states.

All individuals in the population are assumed to be in one of these four states.

The states are: susceptible (S), exposed (E), infected (I) and removed (R).

Comments:

* Those in state R have been infected and either recovered or died.

* Those who have recovered are assumed to have acquired immunity.

* Those in the exposed group are not yet infectious.

Changes in the Infected State
-------------------------------

The flow across states follows the path :math:`S \to E \to I \to R`.


Since we are using a continuum approximation, all individuals in the population are eventually infected when 
the transmission rate is positive and :math:`i(0) > 0`. 

The interest is primarily in 

* the number of infections at a given time (which determines whether or not the health care system is overwhelmed) and
* how long the caseload can be deferred (hopefully until a vaccine arrives)

Assume that there is a constant population of size :math:`N` throughout, then define the proportion of people in each state as :math:`s := S/N` etc.  With this, the SEIR model can be written as 

.. math::
   \begin{aligned} 
        \frac{d s}{d t}  & = - \beta \, s \,  i  
        \\
        \frac{d e}{d t}   & = \beta \,  s \,  i  - \sigma e 
        \\
         \frac{d i}{d t}  & = \sigma  e  - \gamma i
        \\
         \frac{d r}{d t}  & = \gamma  i         
   \end{aligned} 
   :label: seir_system

In these equations,

* :math:`\beta(t)` is called the *transmission rate* (the rate at which individuals bump into others and expose them to the virus).
* :math:`\sigma` is called the *infection rate* (the rate at which those who are exposed become infected)
* :math:`\gamma` is called the *recovery rate* (the rate at which infected people recover or die).
* :math:`dy/dt` represents the time derivative for the particular variable
* Since the states form a partition, so we can reconstruct the "removed" fraction of the population as :math:`r = 1 - s - e - i`.  However, it convenient to :math:`r(t)` in the system for graphing


In addition, we are interested in calculatin the cumulative caseload (i.e., all those who have or have had the infection) as :math:`c = i + r`.  Differentiating that expression and substituing from the time-derivatives of :math:`i(t), r(t)` yields :math:`\frac{d c}{d t} = \sigma e`


Implementing the system of ODEs in :math:`s, e, i` would be enough to implement the model, but we will extend the basic model to enable some policy experiments.

Evolution of Parameters
-----------------------

We will assume that the transmission rate follows a process with a reversion to a value :math:`b` which could conceivably be a policy parameter.  The intuition is that even if the targetted :math:`b(t)` was changed, lags in behavior and implementation would smooth out the transition, where :math:`\eta` governs the speed of :math:`\beta(t)` moves towards :math:`b(t)`. 

.. math::
   \begin{aligned} 
    \frac{d \beta}{d t} &= \eta (b - \beta)
    \end{aligned}

Finally, let :math:`v(t)` be the mortality rate, which we will leave constant for now, i.e. :math:`\frac{d v}{d t} = 0`.  The cumulative deaths can be integrated through the flow :math:`\gamma i` entering the "Removed" state and define the cumulative number of deaths as :math:`m(t)`.  The differential equations then
follow, 

.. math::

    \begin{aligned}\\
    \frac{d v}{d t} &= 0\\
    \frac{d m}{d t} &= v \gamma  i
    \end{aligned}

While we could conveivably integate the total deaths given the solution to the model, it is more convenient to use the integrator built into the ODE solver.  That is, we added :math:`d m(t)/dt` rather than calculating :math:`m(t) = \int_0^t \gamma v(\tau) i(\tau) d \tau` after generating the full :math:`i(t)` path.

This is a common trick when solving systems of ODEs.  While equivalent in principle if you used an appropriate quadrature scheme, this trick becomes especially important and convenient when adaptive time-stepping algorithms are used to solve the ODEs (i.e. there is no fixed time grid).

The system :eq:`seir_system` and the supplemental equations can be written in vector form in terms of the vector :math:`x := (s, e, i, r, c, m, \beta, v)` with parameter vector :math:`p := (\sigma, \gamma, b, \eta)`

.. math::
    \begin{aligned} 
    \frac{d x}{d t} = F(x,t;p) := \begin{bmatrix}
            - \beta \, s \,  i  
        \\
        \beta \,  s \,  i  - \sigma e 
        \\
        \sigma \, e  - \gamma i
        \\
        \gamma \, i
        \\
        \sigma e
        \\
        v \, \gamma \, i
        \\
         \eta (b(t) - \beta)
        \\
        0        
        \end{bmatrix}
    \end{aligned}         
    :label: dfcv

Here note that if :math:`b(t)` is time-invariant, then :math:`F(x)` is time-invariant as well. 

Parameters
----------

Both :math:`\sigma` and :math:`\gamma` are thought of as fixed, biologically determined parameters.

As in Atkeson's note, we set

* :math:`\sigma = 1/5.2` to reflect an average incubation period of 5.2 days.
* :math:`\gamma = 1/18` to match an average illness duration of 18 days.
* :math:`\bar{b} / \gamma = 1.6` to match an **effective reproduction rate** of 1.6, and initially time-invariant
* :math:`v = 0.01` for a one-percent mortality rate

In addition, the transmission rate can be interpreted as 

* :math:`R(t) := \beta(t) / \gamma` where :math:`R(t)` is the *effective reproduction number* at time :math:`t`.

(The notation is standard in the epidemiology literature - though slightly confusing, since :math:`R(t)` is different to
:math:`R`, the symbol that represents the removed state. Throughout the rest of the lecture, we will always use :math:`R` to represent the reproduction number)

As we will initially consider the case where :math:`\beta(0) = \bar{b}`, the value of :math:`\eta` will drop out of this first experiment.

Implementation
==============

First we set the population size to match the US and the parameters as described

.. code-block:: julia

    N = 3.3e8  # US Population
    γ = 1 / 18
    σ = 1 / 5.2
    η = 1 / 20   # a placeholder, drops out of firs texperiments.

Now we construct a function that represents :math:`F` in :eq:`dfcv`

.. code-block:: julia

    # Reminder: could solve dynamics of SEIR states with just first 3 equations
    function F(u, p, t)
        s, e, i, r, c, m, β, v = u
        @unpack σ, γ, b, η = p

        return [-β * s * i;          # ds/dt = -βsi
                 β * s * i -  σ * e; # de/dt =  βsi - σe
                 σ * e - γ * i;      # di/dt =        σe -γi
                 γ * i;              # dr/dt =            γi
                 σ * e;              # dc/dt =        σe
                 v * γ * i;          # dm/dt =           vγi
                 η * (b(t, p) - β);  # dβ/dt = η(b(t) - β)
                 0.0                 # dv/dt = 0
                ]        
    end


The baseline parameters are put into a named tuple generator (see previous lectures using ``Parameters.jl``) with default values discussed above.  

.. code-block:: julia

    (t,p) = p.b̄
    p_gen = @with_kw (T = 550.0, γ = 1.0 / 18, σ = 1 / 5.2, η = 1.0 / 20,
                      b̄ = 1.6 * γ, b = (t, p) -> p.b̄)

Note that the default :math:`b(t)` function is simply the constant function :math:`\bar{b}`


Setting initial conditions, we will assume a fixed :math:`i, e` along with
assuming :math:`r = m = c = 0`, and that :math:`\beta(0) = \bar{b}` and :math:`v(0) = 0.01` 

.. code-block:: julia

    p = p_gen()  # use all default parameters
 
    i_0 = 1e-7
    e_0 = 4.0 * i_0
    s_0 = 1.0 - i_0 - e_0

    u_0 = [s_0, e_0, i_0, 0.0, 0.0, 0.0, p.b̄, 0.01]
    tspan = (0.0, p.T)
    prob = ODEProblem(F, u_0, tspan, p)


The ``tspan`` determines that the :math:`t` used by the sovler, where the scale needs to be consistent with the arrival
rate of the transition probabilities (i.e. the :math:`\gamma, \sigma` were chosen based on daily data).
The time period we investigate will be 550 days, or around 18 months:

Experiments
===========

Let's run some experiments using this code.

First, we can solve the ODE using an appropriate algorthm (e.g. a good default for non-stiff ODEs might be ``Tsit5()``, which is the Tsitouras 5/4 Runge-Kutta method).

Most high-performance ODE solvers appropriate for this class of problems will have adaptive time-stepping, so you
will not specify any sort of grid

.. code-block:: julia

    sol = solve(prob, Tsit5())  # TODO: change the accuracy?
    @show length(sol.t);

We see that the adaptive time-stepping used approximately 45 time-steps to solve this problem to the desires accuracy.  Evaluating the solver at points outside of those time-steps uses the an interpolator consistent with the
solution to the ODE.

See `here <https://docs.sciml.ai/stable/basics/solution/>`__ for details on analyzing the solution, and `here <https://docs.sciml.ai/stable/basics/plot/>`__ for plotting tools.  The built-in plots for the solutions provide all of the `attributes <https://docs.juliaplots.org/latest/tutorial/`__ in `Plots.jl <https://github.com/JuliaPlots/Plots.jl>`__.

.. code_block:: julia

    # TODO: Chris, We could plot something else?  Also labels broken
    plot(sol, vars = [1, 2, 3, 4], label = ["s", "i", "e", "r"], title = "SIER Proportions")



Experiment 1: Constant Reproduction Case
------------------------------

Let's start with the case where :math:`R = \beta / b` is constant.

We calculate the time path of infected people under different assumptions.

.. code-block:: julia
    γ_base = 1.0/18.0
    R_vals = range(1.6, 3.0, length = 6)
    b_vals = R_vals / γ_base
    sols = [solve(ODEProblem(F, u_0, tspan, p_gen(b = b_vals)), Tsit5()) for b in b_vals]
 
    # TODO: Probably clean ways to plot this 

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
successively imposed.

To do this, we will have :math:`\beta(0) \neq b` and examine the dynamics using the :math:`\frac{d \beta}{d t} &= \eta (b - \beta)` differential equation.

.. Mathematica Verification
.. (\[Beta][t] /. 
..     First@DSolve[{\[Beta]'[t] == \[Eta] (b - \[Beta][t]), \[Beta][
..          0] == \[Beta]0}, \[Beta][t], 
..       t] ) == \[Beta]0 E^(-t \[Eta]) + (1 - 
..       E^(-t \[Eta])) b // FullSimplify

      
Note that in the simple case, where :math:`b` is independent of the state, the solution to the ODE with :math:`\beta(0) = \beta_0` is :math:`\beta(t) = \beta_0 e^{-\eta t} + b(1 - e^{-\eta t})`

We will examine the case where :math:`R(t)` starts off at 3 and falls to 1.6 due to the progressive adoption of stricter mitigation measures.

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


The following replicates `additional results <https://drive.google.com/file/d/1uS7n-7zq5gfSgrL3S0HByExmpq4Bn3oh/view>`__ by Andrew Atkeson on the timing of lifting lockdown.

Consider these two mitigation scenarios:

1. :math:`R_t = 0.5` for 30 days and then :math:`R_t = 2` for the remaining 17 months. This corresponds to lifting lockdown in 30 days.

2. :math:`R_t = 0.5` for 120 days and then :math:`R_t = 2` for the remaining 14 months. This corresponds to lifting lockdown in 4 months.

The parameters considered here start the model with 25,000 active infections
and 75,000 agents already exposed to the virus and thus soon to be contagious.

.. code-block:: julia

    # initial conditions
    i_0 = 25_000 / pop_size
    e_0 = 75_000 / pop_size
    s_0 = 1.0 - i_0 - e_0
    x_0 = s_0, e_0, i_0

Let's calculate the paths:

.. code-block:: julia

    #R0_paths = (lambda t: 0.5 if t < 30 else 2,
    #            lambda t: 0.5 if t < 120 else 2)
    #
    #labels = [f'scenario {i}' for i in (1, 2)]
    #
    #i_paths, c_paths = [], []
    #
    #for R0 in R0_paths:
    #    i_path, c_path = solve_path(R0, t_vec, x_init=x_0)
    #    i_paths.append(i_path)
    #    c_paths.append(c_path)


Here is the number of active infections:

.. code-block:: julia

    #plot_paths(i_paths, labels)

What kind of mortality can we expect under these scenarios?

Suppose that 1\% of cases result in death

.. code-block:: julia

    #ν = 0.01

This is the cumulative number of deaths:

.. code-block:: julia

    #paths = [path * ν * pop_size for path in c_paths]
    #plot_paths(paths, labels)

This is the daily death rate:

.. code-block:: julia

    #paths = [path * ν * γ * pop_size for path in i_paths]
    #plot_paths(paths, labels)

Pushing the peak of curve further into the future may reduce cumulative deaths
if a vaccine is found.

