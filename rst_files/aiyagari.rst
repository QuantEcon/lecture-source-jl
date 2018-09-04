.. _aiyagari:

.. include:: /_static/includes/lecture_howto_jl.raw

.. highlight:: julia


*******************************************************
The Aiyagari Model
*******************************************************


Overview
============

In this lecture we describe the structure of a class of models that build on work by Truman Bewley :cite:`Bewley1977`

.. only:: html

    We begin by discussing an example of a Bewley model due to :download:`Rao Aiyagari </_static/pdfs/aiyagari_obit.pdf>`

.. only:: latex

    We begin by discussing an example of a Bewley model due to `Rao Aiyagari <https://lectures.quantecon.org/_downloads/aiyagari_obit.pdf>`__    

The model features 

* Heterogeneous agents 

* A single exogenous vehicle for borrowing and lending

* Limits on amounts individual agents may borrow 


The Aiyagari model has been used to investigate many topics, including

* precautionary savings and the effect of liquidity constraints :cite:`Aiyagari1994`

* risk sharing and asset pricing :cite:`Heaton1996`

* the shape of the wealth distribution :cite:`benhabib2015`

* etc., etc., etc.




References
-------------

The primary reference for this lecture is :cite:`Aiyagari1994`

A textbook treatment is available in chapter 18 of :cite:`Ljungqvist2012`

A continuous time version of the model by SeHyoun Ahn and Benjamin Moll can be found `here <http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/aiyagari_continuous_time.ipynb>`__


The Economy
==============





Households
---------------


Infinitely lived households / consumers face idiosyncratic income shocks 

A unit interval of  *ex ante* identical households face a common borrowing constraint

The savings problem faced by a typical  household is 

.. math::

    \max \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)


subject to

.. math::

    a_{t+1} + c_t \leq w z_t + (1 + r) a_t
    \quad 
    c_t \geq 0, 
    \quad \text{and} \quad
    a_t \geq -B


where

* :math:`c_t` is current consumption

* :math:`a_t` is assets 

* :math:`z_t` is an exogenous component of labor income capturing stochastic unemployment risk, etc.

* :math:`w` is a wage rate

* :math:`r` is a net interest rate

* :math:`B` is the maximum amount that the agent is allowed to borrow

The exogenous process :math:`\{z_t\}` follows a finite state Markov chain with given stochastic matrix :math:`P`

The wage and interest rate are fixed over time 

In this simple version of the model, households supply labor  inelastically because they do not value leisure  



Firms
=======


Firms produce output by hiring capital and labor

Firms act competitively and face constant returns to scale

Since returns to scale are constant the number of firms does not matter 

Hence we can consider a single (but nonetheless competitive) representative firm

The firm's output is

.. math::

    Y_t = A K_t^{\alpha} N^{1 - \alpha}


where 

* :math:`A` and :math:`\alpha` are parameters with :math:`A > 0` and :math:`\alpha \in (0, 1)`

* :math:`K_t` is aggregate capital

* :math:`N` is total labor supply (which is constant in this simple version of the model)


The firm's problem is

.. math::

    max_{K, N} \left\{ A K_t^{\alpha} N^{1 - \alpha} - (r + \delta) K - w N \right\}


The parameter :math:`\delta` is the depreciation rate


From the first-order condition with respect to capital, the firm's inverse demand for capital is

.. math::
    :label: aiy_rgk

    r = A \alpha  \left( \frac{N}{K} \right)^{1 - \alpha} - \delta


Using this expression and the firm's first-order condition for labor, we can pin down
the equilibrium wage rate as a function of :math:`r` as

.. math::
    :label: aiy_wgr

    w(r) = A  (1 - \alpha)  (A \alpha / (r + \delta))^{\alpha / (1 - \alpha)}


Equilibrium
-----------------

We construct  a *stationary rational expectations equilibrium* (SREE)

In such an equilibrium 

* prices induce behavior that generates aggregate quantities consistent with the prices

* aggregate quantities and prices are constant over time


In more detail, an SREE lists a set of prices, savings and production policies such that

* households want to choose the specified savings policies taking the prices as given

* firms maximize profits taking the same prices as given

* the resulting aggregate quantities are consistent with the prices; in particular, the demand for capital equals the supply

* aggregate quantities (defined as cross-sectional averages) are constant 


In practice, once parameter values are set, we can check for an SREE by the following steps

#. pick a proposed quantity :math:`K` for aggregate capital

#. determine corresponding prices, with interest rate :math:`r` determined by :eq:`aiy_rgk` and a wage rate :math:`w(r)` as given in :eq:`aiy_wgr`

#. determine the common optimal savings policy of the households given these prices

#. compute aggregate capital as the mean of steady state capital given this savings policy
   
If this final quantity agrees with :math:`K` then we have a SREE



Code
====

Let's look at how we might compute such an equilibrium in practice

To solve the household's dynamic programming problem we'll use the `DiscreteDP <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/markov/ddp.jl>`_ type from `QuantEcon.jl <http://quantecon.org/julia_index.html>`_

Our first task is the least exciting one: write code that maps parameters for a household problem into the ``R`` and ``Q`` matrices needed to generate an instance of ``DiscreteDP``

Below is a piece of boilerplate code that does just this

In reading the code, the following information will be helpful

* ``R`` needs to be a matrix where ``R[s, a]`` is the reward at state ``s`` under action ``a``

* ``Q`` needs to be a three dimensional array where ``Q[s, a, s']`` is the probability of transitioning to state ``s'`` when the current state is ``s`` and the current action is ``a`` 

(For a detailed discussion of ``DiscreteDP`` see :doc:`this lecture <discrete_dp>`)

Here we take the state to be :math:`s_t := (a_t, z_t)`, where :math:`a_t` is assets and :math:`z_t` is the shock

The action is the choice of next period asset level :math:`a_{t+1}`





The type also includes a default set of parameters that we'll adopt unless otherwise specified

.. code-block:: julia 

    using QuantEcon

    """
    Stores all the parameters that define the household's
    problem.

    ##### Fields

    - `r::Real` : interest rate
    - `w::Real` : wage
    - `σ::Real` : risk aversion
    - `β::AbstractFloat` : discount factor
    - `z_chain::MarkovChain` : MarkovChain for income
    - `a_min::Real` : minimum on asset grid
    - `a_max::Real` : maximum on asset grid
    - `a_size::Integer` : number of points on asset grid
    - `z_size::Integer` : number of points on income grid
    - `n::Integer` : number of points in state space: (a, z)
    - `s_vals::Array{TF} where TF<:AbstractFloat` : stores all the possible (a, z) combinations
    - `s_i_vals::Array{TI} where TI<:Integer` : stores indices of all the possible (a, z) combinations
    - `R::Array{TF} where TF<:AbstractFloat` : reward array
    - `Q::Array{TF} where TF<:AbstractFloat` : transition probability array
    - `u::Function` : utility function
    """
    mutable struct Household{TR<:Real, TF<:AbstractFloat, TI<:Integer}
        r::TR
        w::TR
        σ::TR
        β::TF
        z_chain::MarkovChain{TF, Array{TF, 2}, Array{TF, 1}}
        a_min::TR
        a_max::TR
        a_size::TI
        a_vals::AbstractVector{TF}
        z_size::TI
        n::TI
        s_vals::Array{TF}
        s_i_vals::Array{TI}
        R::Array{TR}
        Q::Array{TR}
        u::Function
    end

    """
    Constructor for `Household`

    ##### Arguments
    - `r::Real(0.01)` : interest rate
    - `w::Real(1.0)` : wage
    - `β::AbstractFloat(0.96)` : discount factor
    - `z_chain::MarkovChain` : MarkovChain for income
    - `a_min::Real(1e-10)` : minimum on asset grid
    - `a_max::Real(18.0)` : maximum on asset grid
    - `a_size::TI(200)` : number of points on asset grid

    """
    function Household{TF<:AbstractFloat}(;
                        r::Real=0.01,
                        w::Real=1.0,
                        σ::Real=1.0,
                        β::TF=0.96,
                        z_chain::MarkovChain{TF,Array{TF,2},Array{TF,1}}
                            =MarkovChain([0.9 0.1; 0.1 0.9], [0.1; 1.0]),
                        a_min::Real=1e-10,
                        a_max::Real=18.0,
                        a_size::Integer=200)

        # set up grids
        a_vals = linspace(a_min, a_max, a_size)
        z_size = length(z_chain.state_values)
        n = a_size*z_size
        s_vals = gridmake(a_vals, z_chain.state_values)
        s_i_vals = gridmake(1:a_size, 1:z_size)

        # set up Q
        Q = zeros(TF, n, a_size, n)
        for next_s_i in 1:n
            for a_i in 1:a_size
                for s_i in 1:n
                    z_i = s_i_vals[s_i, 2]
                    next_z_i = s_i_vals[next_s_i, 2]
                    next_a_i = s_i_vals[next_s_i, 1]
                    if next_a_i == a_i
                        Q[s_i, a_i, next_s_i] = z_chain.p[z_i, next_z_i]
                    end
                end
            end
        end

        if σ == 1   # log utility
            u = x -> log(x)
        else
            u = x -> (x^(1-σ)-1)/(1-σ)
        end

        # placeholder for R
        R = fill(-Inf, n, a_size)
        h = Household(r, w, σ, β, z_chain, a_min, a_max, a_size,
                    a_vals, z_size, n, s_vals, s_i_vals, R, Q, u)

        setup_R!(h, r, w)

        return h

    end

    """
    Update the reward array of a Household object, given
    a new interest rate and wage.

    ##### Arguments
    - `h::Household` : instance of Household type
    - `r::Real(0.01)`: interest rate
    - `w::Real(1.0)` : wage
    """
    function setup_R!(h::Household, r::Real, w::Real)

        # set up R
        R = h.R
        for new_a_i in 1:h.a_size
            a_new = h.a_vals[new_a_i]
            for s_i in 1:h.n
                a = h.s_vals[s_i, 1]
                z = h.s_vals[s_i, 2]
                c = w * z + (1 + r) * a - a_new
                if c > 0
                    R[s_i, new_a_i] = h.u(c)
                end
            end
        end

        h.r = r
        h.w = w
        h.R = R
        h

    end


As a first example of what we can do, let's compute and plot an optimal accumulation policy at fixed prices

.. code-block:: julia 

    using LaTeXStrings
    using Plots
    pyplot()

    # Example prices
    r = 0.03
    w = 0.956

    # Create an instance of Household
    am = Household(a_max=20.0, r=r, w=w)

    # Use the instance to build a discrete dynamic program
    am_ddp = DiscreteDP(am.R, am.Q, am.β)

    # Solve using policy function iteration
    results = solve(am_ddp, PFI)

    # Simplify names
    z_size, a_size  = am.z_size, am.a_size
    z_vals, a_vals = am.z_chain.state_values, am.a_vals
    n = am.n

    # Get all optimal actions across the set of
    # a indices with z fixed in each column
    a_star = reshape([a_vals[results.sigma[s_i]] for s_i in 1:n], a_size, z_size)

    labels = [latexstring("\z = $(z_vals[1])") latexstring("\z = $(z_vals[2])")]
    plot(a_vals, a_star, label=labels, lw=2, alpha=0.6)
    plot!(a_vals, a_vals, label="", color=:black, linestyle=:dash)
    plot!(xlabel="current assets", ylabel="next period assets", grid=false)


The plot shows asset accumulation policies at different values of the exogenous state



Now we want to calculate the equilibrium

Let's do this visually as a first pass

The following code draws aggregate supply and demand curves

The intersection gives equilibrium interest rates and capital


.. code-block:: julia 

    # Firms' parameters
    const A = 1
    const N = 1
    const α = 0.33
    const β = 0.96
    const δ = 0.05

    """
    Compute wage rate given an interest rate, r
    """
    function r_to_w(r::Real)
        return A * (1 - α) * (A * α / (r + δ)) ^ (α / (1 - α))
    end

    """
    Inverse demand curve for capital. The interest rate
    associated with a given demand for capital K.
    """
    function rd(K::Real)
        return A * α * (N / K) ^ (1 - α) - δ
    end

    """
    Map prices to the induced level of capital stock.

    ##### Arguments
    - `am::Household` : Household instance for problem we want to solve
    - `r::Real` : interest rate

    ##### Returns
    - The implied level of aggregate capital
    """
    function prices_to_capital_stock(am::Household, r::Real)

        # Set up problem
        w = r_to_w(r)
        setup_R!(am, r, w)
        aiyagari_ddp = DiscreteDP(am.R, am.Q, am.β)

        # Compute the optimal policy
        results = solve(aiyagari_ddp, PFI)

        # Compute the stationary distribution
        stationary_probs = stationary_distributions(results.mc)[:, 1][1]

        # Return K
        return dot(am.s_vals[:, 1], stationary_probs)
    end

    # Create an instance of Household
    am = Household(β=β, a_max=20.0)

    # Create a grid of r values at which to compute demand and supply of capital
    num_points = 20
    r_vals = linspace(0.005, 0.04, num_points)

    # Compute supply of capital
    k_vals = prices_to_capital_stock.(am, r_vals)

    # Plot against demand for capital by firms
    demand = rd.(k_vals)
    labels =  ["demand for capital" "supply of capital"]
    plot(k_vals, [demand r_vals], label=labels, lw=2, alpha=0.6)
    plot!(xlabel="capital", ylabel="interest rate", xlim=(2, 14), ylim=(0.0, 0.1))


