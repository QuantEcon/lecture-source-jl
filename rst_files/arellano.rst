.. _arellano:

.. include:: /_static/includes/lecture_howto_jl.raw

.. highlight:: julia

***************************************
Default Risk and Income Fluctuations
***************************************

.. contents:: :depth: 2

Overview
============

This lecture computes versions of  Arellano's  :cite:`arellano2008default` model of sovereign default

The model describes interactions among default risk, output,  and  an equilibrium interest rate  that includes a premium for endogenous default risk

The decision maker is a government of a small open economy that borrows from risk-neutral foreign creditors

The foreign lenders must be compensated for default risk

The government borrows and lends abroad in order to  smooth the consumption of its citizens

The government repays its debt only if it wants to, but declining to pay has adverse consequences

The interest rate on government debt adjusts in response to the state-dependent default probability chosen by government

The model yields outcomes that help interpret sovereign default experiences, including

* countercyclical interest rates on sovereign debt

* countercyclical trade balances

* high volatility of consumption relative to output

Notably, long recessions caused by bad draws in the income process increase the government's incentive to default

This can lead to

* spikes in interest rates

* temporary losses of access to international credit markets

* large drops in output, consumption, and welfare

* large capital outflows during recessions

Such dynamics are consistent with experiences of many countries

Structure
===============

In this section we describe the main features of the model

Output, Consumption and Debt
-----------------------------

A small open economy is endowed with an exogenous stochastically fluctuating potential output stream :math:`\{y_t\}`

Potential output is realized only in periods in which the government honors its sovereign debt

The output good can be traded or consumed

The sequence :math:`\{y_t\}` is described by a Markov process with stochastic density kernel :math:`p(y, y')`

Households within the country are identical and rank stochastic consumption streams according to

.. math::
    :label: utility

    \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)

Here

* :math:`0 < \beta < 1` is a time discount factor

* :math:`u` is an increasing and strictly concave utility function

Consumption sequences enjoyed by households are affected by the government's decision to borrow or lend internationally

The government is benevolent in the sense that its aim is to maximize :eq:`utility`

The government is the only domestic actor with access to foreign credit

Because household are averse to consumption fluctuations, the government will try to smooth consumption by borrowing from (and lending to) foreign creditors

Asset Markets
---------------

The only credit instrument available to the government is a one-period bond traded in international credit markets

The bond market has the following features

* The bond matures in one period and is not state contingent

* A purchase of a bond with face value :math:`B'` is a claim to :math:`B'` units of the
  consumption good next period

* To purchase :math:`B'`  next period costs :math:`q B'` now, or, what is equivalent

* For selling :math:`-B'` units of next period goods the seller earns :math:`- q B'` of today's goods

    * if :math:`B' < 0`, then :math:`-q B'` units of the good are received in the current period, for a promise to repay :math:`-B'` units next period

    * there is an equilibrium  price function :math:`q(B', y)` that makes :math:`q` depend on both :math:`B'` and :math:`y`

Earnings on the government portfolio are distributed (or, if negative, taxed) lump sum to households

When the government is not excluded from financial markets, the one-period national budget constraint is

.. math::
    :label: resource

    c = y + B - q(B', y) B'

Here and below, a prime denotes a next period value or a claim maturing next period

To rule out Ponzi schemes, we also require that :math:`B \geq -Z` in every period

* :math:`Z` is chosen to be sufficiently large that the constraint never binds in equilibrium

Financial Markets
-------------------

Foreign creditors

* are risk neutral

* know the domestic output stochastic process :math:`\{y_t\}` and observe :math:`y_t, y_{t-1}, \ldots,` at time :math:`t`

* can borrow or lend without limit in an international credit market at a constant international interest rate :math:`r`

* receive full payment if the government chooses to pay

* receive zero if the government defaults on its one-period debt due

When a government is expected to default next period with  probability :math:`\delta`,  the expected value of a promise to pay one unit of consumption next period is :math:`1 - \delta`.

Therefore, the discounted expected value of a promise to pay :math:`B` next period is

.. math::
    :label: epc

    q = \frac{1 - \delta}{1 + r}

Next we turn to how the government in effect chooses the default probability :math:`\delta`

Government's decisions
-----------------------

At each point in time :math:`t`, the government chooses between

#. defaulting

#. meeting its current obligations and purchasing or selling an optimal quantity of  one-period sovereign debt

Defaulting means declining to repay all of  its current obligations

If the government defaults in the current period, then consumption equals current output

But a sovereign default has two consequences:

#. Output immediately falls from :math:`y` to :math:`h(y)`, where :math:`0 \leq h(y) \leq y`

     * it returns to :math:`y` only after the country regains access to  international credit markets

#. The country loses access to foreign credit markets

Reentering international credit market
---------------------------------------

While in a state of default, the economy regains access to
foreign credit in each subsequent period with probability
:math:`\theta`

Equilibrium
===============

Informally, an equilibrium is a sequence of interest rates on its sovereign debt, a stochastic sequence of government default decisions  and an implied flow of household consumption such that

#. Consumption and assets satisfy the national budget constraint

#. The government maximizes household utility taking into account

   * the resource constraint

   * the effect of its choices on the price of bonds

   * consequences of defaulting now for future net output and future borrowing and lending opportunities

#. The interest rate on the government's debt includes a risk-premium sufficient to make foreign creditors expect on average to earn the constant risk-free international interest rate


To express these ideas more precisely, consider first the choices of the
government, which

#. enters a period with initial assets :math:`B`, or  what is the same thing, initial debt to be repaid now of :math:`-B`

#. observes current output :math:`y`, and

#. chooses either

    #. to default, or

    #. to pay  :math:`-B` and set next period's debt due to :math:`-B'`

In a  recursive formulation,

* state variables for the government comprise the pair :math:`(B, y)`

* :math:`v(B, y)` is the optimum value of the government's problem when at the beginning of a period  it faces the choice of whether to honor or default

* :math:`v_c(B, y)` is the value of choosing to pay obligations falling due

* :math:`v_d(y)` is the value of choosing to default

:math:`v_d(y)` does not depend on :math:`B` because, when access to credit is eventually regained, net foreign assets equal :math:`0`

Expressed recursively, the value of defaulting is

.. math::

    v_d(y) = u(h(y))
        + \beta \int \left\{
                \theta v(0, y') + (1 - \theta) v_d(y')
                \right\}
                p(y, y') dy'

The value of paying is

.. math::

    v_c(B, y) = \max_{B' \geq -Z}
           \left\{
                u(y - q(B', y) B' + B)
                + \beta \int v(B', y') p(y, y') dy'
          \right\}

The three value functions are linked by

.. math::

    v(B, y) = \max\{ v_c(B, y), v_d(y) \}

The government chooses to default when

.. math::

    v_c(B, y) < v_d(y)

and hence given :math:`B'` the probability of default next period  is

.. math::
    :label: delta

    \delta(B', y) := \int \mathbb 1\{v_c(B', y') < v_d(y') \} p(y, y') dy'

Given zero profits for foreign creditors in equilibrium, we can combine :eq:`epc` and :eq:`delta` to pin down
the bond price function:

.. math::
    :label: bondprice

    q(B', y) = \frac{1 - \delta(B', y)}{1 + r}

Definition of equilibrium
---------------------------

An *equilibrium* is

* a  pricing function :math:`q(B',y)`,

* a triple of value functions :math:`(v_c(B, y), v_d(y), v(B,y))`,

* a decision rule telling the government when to default and when to pay as a function of the state  :math:`(B, y)`, and

* an asset accumulation rule that, conditional on choosing not to  default,  maps :math:`(B,y)` into :math:`B'`

such that

* The three Bellman equations for :math:`(v_c(B, y), v_d(y), v(B,y))` are satisfied

* Given the price function :math:`q(B',y)`, the default decision rule and the asset accumulation decision rule attain the optimal value function  :math:`v(B,y)`, and

* The price function :math:`q(B',y)` satisfies equation :eq:`bondprice`

Computation
===============

Let's now compute an equilibrium of Arellano's model

The equilibrium objects are the value function :math:`v(B, y)`, the associated default decision rule, and the pricing function :math:`q(B', y)`

We'll use our code to replicate Arellano's results

After that we'll perform some additional simulations

The majority of the code below was written by `Chase Coleman <https://github.com/cc7768>`_

It uses a slightly modified version of the algorithm recommended by Arellano

* The appendix to :cite:`arellano2008default` recommends
  value function iteration until convergence, updating the price, and then repeating

* Instead, we update the bond price at every value function iteration step

The second approach is faster and the two different procedures deliver very similar results

Here is a more detailed description of our algorithm:

1. Guess a value function :math:`v(B, y)` and price function :math:`q(B', y)`

2. At each pair :math:`(B, y)`,

   * update the value of defaulting :math:`v_d(y)`

   * update the value of continuing :math:`v_c(B, y)`

3.  Update the value function `v(B, y)`, the default rule,  the implied ex ante default probability,  and the price function

4. Check for convergence. If converged, stop. If not, go to step 2.

We use simple discretization on a grid of asset holdings and income levels

.. note that the next line will need to be changed when we add a Julia version

The output process is discretized using `Tauchen's quadrature method <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/markov/markov_approx.jl>`_

The code can be found below:

(Results and discussion follow the code)

.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

    using QuantEcon

    struct ArellanoEconomy{TF<:AbstractFloat, TI<:Integer}
        # Model Parameters
        β::TF
        γ::TF
        r::TF
        ρ::TF
        η::TF
        θ::TF

        # Grid Parameters
        ny::TI
        nB::TI
        ygrid::Vector{TF}
        ydefgrid::Vector{TF}
        Bgrid::Vector{TF}
        Π::Matrix{TF}

        # Value function
        vf::Matrix{TF}
        vd::Matrix{TF}
        vc::Matrix{TF}
        policy::Matrix{TF}
        q::Matrix{TF}
        defprob::Matrix{TF}
    end

    function ArellanoEconomy(;β::TF = .953,
                              γ::TF = 2.,
                              r::TF = 0.017,
                              ρ::TF = 0.945,
                              η::TF = 0.025,
                              θ::TF = 0.282,
                              ny::Integer = 21,
                              nB::Integer = 251) where TF<:AbstractFloat

        # Create grids
        Bgrid = collect(range(-.4, stop = .4, length = nB))
        mc = tauchen(ny, ρ, η)
        Π = mc.p
        ygrid = exp.(mc.state_values)
        ydefgrid = min.(.969 * mean(ygrid), ygrid)

        # Define value functions (Notice ordered different than Python to take
        # advantage of column major layout of Julia)
        vf = zeros(nB, ny)
        vd = zeros(1, ny)
        vc = zeros(nB, ny)
        policy = zeros(TF, nB, ny)
        q = ones(nB, ny) .* (1 / (1 + r))
        defprob = zeros(TF, nB, ny)

        return ArellanoEconomy(β, γ, r, ρ, η, θ, ny, nB, ygrid, ydefgrid, Bgrid, Π,
                                vf, vd, vc, policy, q, defprob)
    end

    u(ae::ArellanoEconomy, c) = c^(1 - ae.γ) / (1 - ae.γ)
    _unpack(ae::ArellanoEconomy) =
        ae.β, ae.γ, ae.r, ae.ρ, ae.η, ae.θ, ae.ny, ae.nB
    _unpackgrids(ae::ArellanoEconomy) =
        ae.ygrid, ae.ydefgrid, ae.Bgrid, ae.Π, ae.vf, ae.vd, ae.vc, ae.policy, ae.q, ae.defprob


    function one_step_update!(ae::ArellanoEconomy,
                              EV::Matrix{TF},
                              EVd::Matrix{TF},
                              EVc::Matrix{TF}) where TF<:AbstractFloat

        # Unpack stuff
        β, γ, r, ρ, η, θ, ny, nB = _unpack(ae)
        ygrid, ydefgrid, Bgrid, Π, vf, vd, vc, policy, q, defprob = _unpackgrids(ae)
        zero_ind = searchsortedfirst(Bgrid, 0.)

        for iy in 1:ny
            y = ae.ygrid[iy]
            ydef = ae.ydefgrid[iy]

            # Value of being in default with income y
            defval = u(ae, ydef) + β*(θ*EVc[zero_ind, iy] + (1-θ)*EVd[1, iy])
            ae.vd[1, iy] = defval

            for ib in 1:nB
                B = ae.Bgrid[ib]

                current_max = -1e14
                pol_ind = 0
                for ib_next=1:nB
                    c = max(y - ae.q[ib_next, iy]*Bgrid[ib_next] + B, 1e-14)
                    m = u(ae, c) + β * EV[ib_next, iy]

                    if m > current_max
                        current_max = m
                        pol_ind = ib_next
                    end

                end

                # Update value and policy functions
                ae.vc[ib, iy] = current_max
                ae.policy[ib, iy] = pol_ind
                ae.vf[ib, iy] = defval > current_max ? defval : current_max
            end
        end
    end

    function compute_prices!(ae)
        # Unpack parameters
        β, γ, r, ρ, η, θ, ny, nB = _unpack(ae)

        # Create default values with a matching size
        vd_compat = repeat(ae.vd, nB)
        default_states = vd_compat .> ae.vc

        # Update default probabilities and prices
        copyto!(ae.defprob, default_states * ae.Π')
        copyto!(ae.q, (1 .- ae.defprob) / (1 + r))
        return
    end

    function vfi!(ae; tol = 1e-8, maxit = 10000)

        # Unpack stuff
        β, γ, r, ρ, η, θ, ny, nB = _unpack(ae)
        ygrid, ydefgrid, Bgrid, Π, vf, vd, vc, policy, q, defprob = _unpackgrids(ae)
        Πt = Π'

        # Iteration stuff
        it = 0
        dist = 10.

        # Allocate memory for update
        V_upd = similar(ae.vf)

        while dist > tol && it < maxit
            it += 1

            # Compute expectations for this iterations
            # (We need Π' because of order value function dimensions)
            copyto!(V_upd, ae.vf)
            EV = ae.vf * Πt
            EVd = ae.vd * Πt
            EVc = ae.vc * Πt

            # Update Value Function
            one_step_update!(ae, EV, EVd, EVc)

            # Update prices
            compute_prices!(ae)

            dist = maximum(abs, V_upd - ae.vf)

            if it%25 == 0
                println("Finished iteration $(it) with dist of $(dist)")
            end
        end
    end

    function QuantEcon.simulate(ae,
                                capT::TI = 5000;
                                y_init::TF = mean(ae.ygrid),
                                B_init::TF = mean(ae.Bgrid)
                                ) where {TI<:Integer, TF<:AbstractFloat}

        # Get initial indices
        zero_index = searchsortedfirst(ae.Bgrid, 0.)
        y_init_ind = searchsortedfirst(ae.ygrid, y_init)
        B_init_ind = searchsortedfirst(ae.Bgrid, B_init)

        # Create a QE MarkovChain
        mc = MarkovChain(ae.Π)
        y_sim_indices = simulate(mc, capT+1; init=y_init_ind)

        # Allocate and Fill output
        y_sim_val = zeros(TF, capT+1)
        B_sim_val, q_sim_val = similar(y_sim_val), similar(y_sim_val)
        B_sim_indices = zeros(TI, capT+1)
        default_status = fill(false, capT+1)
        B_sim_indices[1], default_status[1] = B_init_ind, false
        y_sim_val[1], B_sim_val[1] = ae.ygrid[y_init_ind], ae.Bgrid[B_init_ind]

        for t in 1:capT
            # Get today's indexes
            yi, Bi = y_sim_indices[t], B_sim_indices[t]
            defstat = default_status[t]

            # If you are not in default
            if !defstat
                default_today = ae.vc[Bi, yi] < ae.vd[yi] ? true : false

                if default_today
                    # Default values
                    default_status[t] = true
                    default_status[t+1] = true
                    y_sim_val[t] = ae.ydefgrid[y_sim_indices[t]]
                    B_sim_indices[t+1] = zero_index
                    B_sim_val[t+1] = 0.
                    q_sim_val[t] = ae.q[zero_index, y_sim_indices[t]]
                else
                    default_status[t] = false
                    y_sim_val[t] = ae.ygrid[y_sim_indices[t]]
                    B_sim_indices[t+1] = ae.policy[Bi, yi]
                    B_sim_val[t+1] = ae.Bgrid[B_sim_indices[t+1]]
                    q_sim_val[t] = ae.q[B_sim_indices[t+1], y_sim_indices[t]]
                end

            # If you are in default
            else
                B_sim_indices[t+1] = zero_index
                B_sim_val[t+1] = 0.
                y_sim_val[t] = ae.ydefgrid[y_sim_indices[t]]
                q_sim_val[t] = ae.q[zero_index, y_sim_indices[t]]

                # With probability θ exit default status
                if rand() < ae.θ
                    default_status[t+1] = false
                else
                    default_status[t+1] = true
                end
            end
        end

        return (y_sim_val[1:capT], B_sim_val[1:capT], q_sim_val[1:capT],
                default_status[1:capT])
    end

Results
===============

Let's start by trying to replicate the results obtained in :cite:`arellano2008default`

In what follows, all results are computed using Arellano's parameter values

The values can be seen in the function `ArellanoEconomy` shown above

* For example, `r=0.017` matches the average quarterly rate on a 5 year US treasury
  over the period 1983--2001

Details on how to compute the figures are reported as solutions to the exercises

The first figure shows the bond price schedule and replicates Figure 3 of Arellano, where :math:`y_L` and :math:`Y_H` are particular below average and above average values of output :math:`y`

.. figure:: /_static/figures/arellano_bond_prices.png

* :math:`y_L` is 5% below the mean of the :math:`y` grid values

* :math:`y_H` is 5% above  the mean of the :math:`y` grid values

The grid used to compute this figure was relatively coarse (`ny, nB = 21, 251`) in order to match Arrelano's findings

Here's the same relationships computed on a finer grid (`ny, nB = 51, 551`)

.. figure:: /_static/figures/arellano_bond_prices_2.png

In either case, the figure shows that

* Higher levels of debt (larger :math:`-B'`) induce larger discounts on the face value, which correspond to  higher interest rates

* Lower income also causes more discounting, as foreign creditors anticipate greater likelihood of default

The next figure plots value functions and replicates the right hand panel of Figure 4 of :cite:`arellano2008default`

.. figure:: /_static/figures/arellano_value_funcs.png

We can use the results of the computation to study the default probability
:math:`\delta(B', y)` defined in :eq:`delta`

The next plot shows these default probabilities over :math:`(B', y)` as a heat
map

.. figure:: /_static/figures/arellano_default_probs.png

As anticipated, the probability that the government chooses to default in the
following period increases with indebtedness and falls with income

Next let's run a time series simulation of :math:`\{y_t\}`, :math:`\{B_t\}` and :math:`q(B_{t+1}, y_t)`

The grey vertical bars correspond to periods when the economy is excluded from financial markets because of a past default

.. figure:: /_static/figures/arellano_time_series.png

One notable feature of the simulated data is the nonlinear response of interest rates

Periods of relative stability are followed by sharp spikes in the discount rate on government debt

Exercises
==============

.. _arellano_ex1:

Exercise 1
------------

To the extent that you can, replicate the figures shown above

* Use the parameter values listed as defaults in the function `ArellanoEconomy`

* The time series will of course vary depending on the shock draws

Solutions
==========

.. code-block:: julia

    using Gadfly, Compose, ColorTypes, DataFrames

Compute the value function, policy and equilibrium prices

.. code-block:: julia

    ae = ArellanoEconomy(β = .953,     # time discount rate
                         γ = 2.,       # risk aversion
                         r = 0.017,    # international interest rate
                         ρ = .945,     # persistence in output
                         η = 0.025,    # st dev of output shock
                         θ = 0.282,    # prob of regaining access
                         ny = 21,      # number of points in y grid
                         nB = 251)     # number of points in B grid

    # now solve the model on the grid.
    vfi!(ae)

.. code-block:: julia
  :class: test

  @testset begin
      @test ae.ygrid[5] ≈ 0.871460195968125
      @test ae.Bgrid[5] ≈ -0.3872
  end

Compute the bond price schedule as seen in figure 3 of Arellano (2008)

.. code-block:: julia

    # Create "Y High" and "Y Low" values as 5% devs from mean
    high, low = mean(ae.ygrid)*1.05, mean(ae.ygrid)*.95
    iy_high, iy_low = map(x->searchsortedfirst(ae.ygrid, x), (high, low))

    # Extract a suitable plot grid
    x = Float64[]
    q_low = Float64[]
    q_high = Float64[]
    for i in 1:ae.nB
        b = ae.Bgrid[i]
        if -0.35 ≤ b ≤ 0  # To match fig 3 of Arellano
            push!(x, b)
            push!(q_low, ae.q[i, iy_low])
            push!(q_high, ae.q[i, iy_high])
        end
    end

    # generate plot
    p1=plot(x=repeat(x, outer=[2]), y=[q_low; q_high],
         color=repeat([:Low, :High], inner=[length(x)]),
         Guide.title("Bond price schedule q(y, B')"),
         Guide.xlabel("B'"), Guide.ylabel("q"),
         Guide.colorkey(title = "y"), Geom.line)

.. code-block:: julia 
  :class: test 

  @testset begin 
    @test q_high[4] ≈ 0.2664149144450229
    @test q_low[20] ≈ 2.2418853380628388e-5
    @test x[17] == -0.2976
  end 

Draw a plot of the value functions

.. code-block:: julia

    p2=plot(x=repeat(ae.Bgrid, outer=[2]),
         y=vec(ae.vf[:, [iy_low, iy_high]]),
         color=repeat([:Low, :High], inner=[length(ae.Bgrid)]),
         Guide.title("Value functions"),
         Guide.xlabel("B"), Guide.ylabel("V(y,B)"),
         Guide.colorkey(title = "y"), Geom.line)

Draw a heat map for default probability

.. code-block:: julia

    p3=plot(x_min=repeat(ae.Bgrid[1:end-1], inner=[ae.ny-1]),
         x_max=repeat(ae.Bgrid[2:end], inner=[ae.ny-1]),
         y_min=repeat(ae.ygrid[1:end-1], outer=[ae.nB-1]),
         y_max=repeat(ae.ygrid[2:end], outer=[ae.nB-1]),
         x=(repeat(ae.Bgrid[1:end-1], inner=[ae.ny-1]) + repeat(ae.Bgrid[2:end], inner=[ae.ny-1]))/2,
         y=(repeat(ae.ygrid[1:end-1], outer=[ae.nB-1]) + repeat(ae.ygrid[2:end], outer=[ae.nB-1]))/2,
         color=clamp.(vec(ae.defprob[1:end-1, 1:end-1]'), 0, 1),
         Geom.rectbin,
         Guide.xlabel("B'"), Guide.ylabel("y"),
         Guide.title("Probability of default"), Geom.rectbin,
         Scale.y_continuous(minvalue=0.8, maxvalue=1.2),
         Scale.x_continuous(minvalue=minimum(ae.Bgrid), maxvalue=0.0),
         Scale.color_continuous(minvalue=0, maxvalue=1))

Plot a time series of major variables simulated from the model

.. code-block:: julia

    using Random
    # set random seed for consistent result
    Random.seed!(348938)

    # simulate
    T = 250
    y_vec, B_vec, q_vec, default_vec = simulate(ae, T)

    # find starting and ending periods of recessions
    defs = findall(default_vec)
    def_breaks = diff(defs) .> 1
    def_start = defs[[true; def_breaks]]
    def_end = defs[[def_breaks; true]]

    # construct boxes that shade periods of default
    def_box = Guide.annotation(compose(context(),
                                       [rectangle(i[1], 0h, i[2]-i[1], 1h)
                                        for i=zip(def_start, def_end)]...,
                                      fill(RGBA(0.5, 0.5, 0.5, 0.2))))

    # xy labels are common for all plots
    xy_lab = [Guide.xlabel("time"), Guide.ylabel("")]

    # now iterate over three variables and put them into an array
    p4 = Gadfly.Plot[]
    for (vec, name) in [(y_vec, "Output"), (B_vec, "Foreign assets"), (q_vec, "Bond price")]
        push!(p4,
              plot(x=1:T, y=vec, Geom.line, def_box, Guide.title(name), xy_lab...))
    end

    # set final plot height and vertically stack the above three plots
    set_default_plot_size(6inch, 8inch)
    vstack(p4...)

.. code-block:: julia 
  :class: test 

  @testset begin 
    @test def_end == [62, 157, 198]
    @test def_start == [60, 154, 196]
    @test def_breaks == Bool[false, false, true, false, false, false, true, false, false]
    @test defs == [60, 61, 62, 154, 155, 156, 157, 196, 197, 198]
    @test y_vec[4] ≈ 1.0712139563752547
    @test B_vec[40] == -0.0768
    @test q_vec[140] ≈ 0.9738927780828847
    @test default_vec[240] == false
  end 
