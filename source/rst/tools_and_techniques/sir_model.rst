.. include:: /_static/includes/header.raw

.. highlight:: julia

************************************
:index:`Modeling COVID 19`
************************************

.. contents:: :depth: 2



Overview
=============

This is a Julia version of the code for analyzing the COVID-19 pandemic
provided by `Andrew Atkeson <https://sites.google.com/site/andyatkeson/>`__.

See, in particular

* `NBER Working Paper No. 26867 <https://www.nber.org/papers/w26867>`__ 
* `COVID-19 Working papers and code <https://sites.google.com/site/andyatkeson/home?authuser=0>`__


The purpose of his notes is to introduce economists to quantitative modeling
of infectious disease dynamics.

Dynamics are modeled using a standard SIR (Susceptible-Infected-Removed) model
of disease spread.

The model dynamics are represented by a system of ordinary differential
equations when the number of agents is large and there are no exogenous stochastic shocks.

In addition, we will introduce stochastic differential equations (SDE) and the connection
to the Langevin equations typically used in the approximation of chemical reactions in well-mixed.

The main objective is to study the impact of suppression through social
distancing on the spread of the infection.

The focus is on US outcomes but the parameters can be adjusted to study
other countries.

Also, variations on the classic SIR model in Julia  `here <https://github.com/epirecipes/sir-julia>`__.

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics, Random, SparseArrays

.. code-block:: julia
    :class: Test

    using Test # Put this before any code in the lecture.

In addition, we will be exploring packages within the `SciML ecosystem <https://github.com/SciML/>`__.

.. code-block:: julia

    using OrdinaryDiffEq, StochasticDiffEq, ModelingToolkit


The SIR Model
=============

In the version of the SIR model we will analyze there are four states.

All individuals in the population are assumed to be in one of these four states.

The states are: susceptible (S), exposed (E), infected (I) and removed (R).

Comments:

* Those in state R have been infected and either recovered or died.

* Those who have recovered are assumed to have acquired immunity.

* Those in the exposed group are not yet infectious.

Time Path
----------

The flow across states follows the path :math:`S \to E \to I \to R`.


All individuals in the population are eventually infected when 
the transmission rate is positive and :math:`i(0) > 0`. 

The interest is primarily in 

* the number of infections at a given time (which determines whether or not the health care system is overwhelmed) and
* how long the caseload can be deferred (hopefully until a vaccine arrives)

Using lower case letters for the fraction of the population in each state, the
dynamics are

.. math::
   \begin{aligned} 
        \dot s(t)  & = - \beta(t) \, s(t) \,  i(t)  
        \\
        \dot e(t)  & = \beta(t) \,  s(t) \,  i(t)  - σ e(t) 
        \\
        \dot i(t)  & = σ e(t)  - γ i(t) 
   \end{aligned} 
   :label: sir_system

In these equations,

* :math:`\beta(t)` is called the *transmission rate* (the rate at which individuals bump into others and expose them to the virus).
* :math:`\sigma` is called the *infection rate* (the rate at which those who are exposed become infected)
* :math:`\gamma` is called the *recovery rate* (the rate at which infected people recover or die).
* the dot symbol :math:`\dot y` represents the time derivative :math:`dy/dt`.


We do not need to model the fraction :math:`r` of the population in state :math:`R` separately because the states form a partition.

In particular, the "removed" fraction of the population is :math:`r = 1 - s - e - i`.

We will also track :math:`c = i + r`, which is the cumulative caseload
(i.e., all those who have or have had the infection).

The system :eq:`sir_system` can be written in vector form as

.. math::
    \dot x = F(x, t),  \qquad x := (s, e, i)
    :label: dfcv

for suitable definition of :math:`F` (see the code below).


Parameters
----------

Both :math:`\sigma` and :math:`\gamma` are thought of as fixed, biologically determined parameters.

As in Atkeson's note, we set

* :math:`\sigma = 1/5.2` to reflect an average incubation period of 5.2 days.
* :math:`\gamma = 1/18` to match an average illness duration of 18 days.

The transmission rate is modeled as 

* :math:`\beta(t) := R(t) \gamma` where :math:`R(t)` is the *effective reproduction number* at time :math:`t`.

(The notation is slightly confusing, since :math:`R(t)` is different to
:math:`R`, the symbol that represents the removed state.)


Implementation
==============

First we set the population size to match the US.

.. code-block:: julia

    pop_size = 3.3e8

Next we fix parameters as described above.

.. code-block:: julia

    γ = 1 / 18
    σ = 1 / 5.2

Now we construct a function that represents :math:`F` in :eq:`dfcv`

.. code-block:: julia

    #def F(x, t, R0=1.6):
    #    """
    #    Time derivative of the state vector.
    #
    #        * x is the state vector (array_like)
    #        * t is time (scalar)
    #        * R0 is the effective transmission rate, defaulting to a constant
    #
    #    """
    #    s, e, i = x
    #
    #    # New exposure of susceptibles
    #    β = R0(t) * γ if callable(R0) else R0 * γ
    #    ne = β * s * i   
    #    
    #    # Time derivatives
    #    ds = - ne
    #    de = ne - σ * e
    #    di = σ * e - γ * i
    #    
    #    return ds, de, di

Note that ``R0`` can be either constant or a given function of time.

The initial conditions are set to

.. code-block:: julia

    # initial conditions of s, e, i
    i_0 = 1e-7
    e_0 = 4.0 * i_0
    s_0 = 1.0 - i_0 - e_0

In vector form the initial condition is 

.. code-block:: julia

    x_0 = s_0, e_0, i_0

We solve for the time path numerically using `odeint`, at a sequence of dates
``t_vec``.

.. code-block:: julia

    #def solve_path(R0, t_vec, x_init=x_0):
    #    """
    #    Solve for i(t) and c(t) via numerical integration, 
    #    given the time path for R0.
    #
    #    """
    #    G = lambda x, t: F(x, t, R0)
    #    s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()
    #
    #    c_path = 1 - s_path - e_path       # cumulative cases
    #    return i_path, c_path



Experiments
===========

Let's run some experiments using this code.

The time period we investigate will be 550 days, or around 18 months:

.. code-block:: julia

    t_length = 550
    grid_size = 1000
    #t_vec = np.linspace(0, t_length, grid_size)



Experiment 1: Constant R0 Case
------------------------------


Let's start with the case where ``R0`` is constant.

We calculate the time path of infected people under different assumptions for ``R0``:

.. code-block:: julia

    #R0_vals = np.linspace(1.6, 3.0, 6)
    #labels = [f'$R0 = {r:.2f}$' for r in R0_vals]
    #i_paths, c_paths = [], []
    #
    #for r in R0_vals:
    #    i_path, c_path = solve_path(r, t_vec)
    #    i_paths.append(i_path)
    #    c_paths.append(c_path)

Here's some code to plot the time paths.

.. code-block:: julia

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

Here's a specification for ``R0`` as a function of time.

.. code-block:: julia

    #def R0_mitigating(t, r0=3, η=1, r_bar=1.6): 
    #    R0 = r0 * exp(- η * t) + (1 - exp(- η * t)) * r_bar
    #    return R0

The idea is that ``R0`` starts off at 3 and falls to 1.6.

This is due to progressive adoption of stricter mitigation measures.

The parameter ``η`` controls the rate, or the speed at which restrictions are
imposed.

We consider several different rates:

.. code-block:: julia

    #η_vals = 1/5, 1/10, 1/20, 1/50, 1/100
    #labels = [fr'$\eta = {η:.2f}$' for η in η_vals]

This is what the time path of ``R0`` looks like at these alternative rates:

.. code-block:: julia

    #fig, ax = plt.subplots()
    #
    #for η, label in zip(η_vals, labels):
    #    ax.plot(t_vec, R0_mitigating(t_vec, η=η), label=label)
    #
    #ax.legend()
    #plt.show()

Let's calculate the time path of infected people:

.. code-block:: julia

    #i_paths, c_paths = [], []
    #
    #for η in η_vals:
    #    R0 = lambda t: R0_mitigating(t, η=η)
    #    i_path, c_path = solve_path(R0, t_vec)
    #    i_paths.append(i_path)
    #    c_paths.append(c_path)


This is current cases under the different scenarios:

.. code-block:: julia

    #plot_paths(i_paths, labels)

Here are cumulative cases, as a fraction of population:

.. code-block:: julia

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


