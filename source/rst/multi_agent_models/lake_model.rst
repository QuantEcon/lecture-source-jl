.. _lake_model:

.. include:: /_static/includes/header.raw

.. highlight:: julia

*******************************************
A Lake Model of Employment and Unemployment
*******************************************

.. index::
    single: Lake Model

.. contents:: :depth: 2

Overview
========

This lecture describes what has come to be called a *lake model*.

The lake model is a basic tool for modeling unemployment.

It allows us to analyze

* flows between unemployment and employment

* how these flows influence steady state employment and unemployment rates

It is a good model for interpreting monthly labor department reports on gross and net jobs created and jobs destroyed.

The "lakes" in the model are the pools of employed and unemployed.

The "flows" between the lakes are caused by

* firing and hiring

* entry and exit from the labor force

For the first part of this lecture, the parameters governing transitions into
and out of unemployment and employment are exogenous.

Later, we'll determine some of these transition rates endogenously using the :doc:`McCall search model <../dynamic_programming/mccall_model>`.

We'll also use some nifty concepts like ergodicity, which provides a fundamental link between *cross-sectional* and *long run time series* distributions.

These concepts will help us build an equilibrium model of ex ante homogeneous workers whose different luck generates variations in their ex post experiences.

Prerequisites
-------------

Before working through what follows, we recommend you read the :doc:`lecture
on finite Markov chains <../tools_and_techniques/finite_markov>`.

You will also need some basic :doc:`linear algebra <../tools_and_techniques/linear_algebra>` and probability.

The Model
=========

The economy is inhabited by a very large number of ex ante identical workers.

The workers live forever, spending their lives moving between unemployment and employment.

Their rates of  transition between employment and unemployment are  governed by the following parameters:

* :math:`\lambda`, the job finding rate for currently unemployed workers

* :math:`\alpha`, the dismissal rate for currently employed workers

* :math:`b`, the entry rate into the labor force

* :math:`d`, the exit rate from the labor force

The growth rate of the labor force evidently equals :math:`g=b-d`.

Aggregate Variables
-------------------

We want to derive the dynamics of the following aggregates

* :math:`E_t`, the total number of employed workers at date :math:`t`

* :math:`U_t`, the total number of unemployed workers at :math:`t`

* :math:`N_t`, the number of workers in the labor force at :math:`t`

We also want to know the values of the following objects

* The employment rate :math:`e_t := E_t/N_t`.

* The unemployment rate :math:`u_t := U_t/N_t`.

(Here and below, capital letters represent stocks and lowercase letters represent flows)

Laws of Motion for Stock Variables
----------------------------------

We begin by constructing laws of motion for the aggregate variables :math:`E_t,U_t, N_t`.

Of the mass of workers :math:`E_t` who are employed at date :math:`t`,

* :math:`(1-d)E_t` will remain in the labor force

* of these, :math:`(1-\alpha)(1-d)E_t` will remain employed

Of the mass of workers :math:`U_t` workers who are currently unemployed,

* :math:`(1-d)U_t` will remain in the labor force

* of these, :math:`(1-d) \lambda U_t` will become employed

Therefore,  the number of workers who will be employed at date :math:`t+1` will be

.. math::

    E_{t+1} = (1-d)(1-\alpha)E_t + (1-d)\lambda U_t

A similar analysis implies

.. math::

    U_{t+1} = (1-d)\alpha E_t + (1-d)(1-\lambda)U_t + b (E_t+U_t)

The value :math:`b(E_t+U_t)` is the mass of new workers entering the labor force unemployed.

The total stock of workers :math:`N_t=E_t+U_t` evolves as

.. math::

    N_{t+1} = (1+b-d)N_t = (1+g)N_t

Letting :math:`X_t := \left(\begin{matrix}U_t\\E_t\end{matrix}\right)`, the law of motion for :math:`X`  is

.. math::

    X_{t+1} = A X_t
    \quad \text{where} \quad
    A :=
    \begin{pmatrix}
        (1-d)(1-\lambda) + b & (1-d)\alpha + b  \\
        (1-d)\lambda & (1-d)(1-\alpha)
    \end{pmatrix}

This law tells us how total employment and unemployment evolve over time.

Laws of Motion for Rates
------------------------

Now let's derive the law of motion for rates.

To get these we can divide both sides of :math:`X_{t+1} = A X_t` by  :math:`N_{t+1}` to get

.. math::

    \begin{pmatrix}
        U_{t+1}/N_{t+1} \\
        E_{t+1}/N_{t+1}
    \end{pmatrix}
    =
    \frac1{1+g} A
    \begin{pmatrix}
        U_{t}/N_{t}
        \\
        E_{t}/N_{t}
    \end{pmatrix}

Letting

.. math::

    x_t :=
    \left(\begin{matrix}
        u_t\\ e_t
    \end{matrix}\right)
    = \left(\begin{matrix}
        U_t/N_t\\ E_t/N_t
    \end{matrix}\right)

we can also write this as

.. math::

    x_{t+1} = \hat A x_t
    \quad \text{where} \quad
    \hat A := \frac{1}{1 + g} A

You can check that :math:`e_t + u_t = 1` implies that :math:`e_{t+1}+u_{t+1} = 1`.

This follows from the fact that the columns of :math:`\hat A` sum to 1.

Implementation
==============

Let's code up these equations.

Here's the code:

Setup
-----

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: hide-output

    using LinearAlgebra, Statistics
    using Distributions, Expectations, NLsolve, Parameters, Plots
    using QuantEcon, Roots, Random

.. code-block:: julia
    :class: test

    using Test

.. code-block:: julia

    gr(fmt = :png);

.. code-block:: julia

    LakeModel = @with_kw (λ = 0.283, α = 0.013, b = 0.0124, d = 0.00822)

    function transition_matrices(lm)
        @unpack λ, α, b, d = lm
        g = b - d
        A = [(1 - λ) * (1 - d) + b      (1 - d) * α + b
            (1 - d) * λ                 (1 - d) * (1 - α)]
        Â = A ./ (1 + g)
        return (A = A, Â = Â)
    end

    function rate_steady_state(lm)
        @unpack Â = transition_matrices(lm)
        sol = fixedpoint(x -> Â * x, fill(0.5, 2))
        converged(sol) || error("Failed to converge in $(result.iterations) iterations")
        return sol.zero
    end

    function simulate_stock_path(lm, X0, T)
        @unpack A = transition_matrices(lm)
        X_path = zeros(eltype(X0), 2, T)
        X = copy(X0)
        for t in 1:T
            X_path[:, t] = X
            X = A * X
        end
        return X_path
    end

    function simulate_rate_path(lm, x0, T)
        @unpack Â = transition_matrices(lm)
        x_path = zeros(eltype(x0), 2, T)
        x = copy(x0)
        for t in 1:T
            x_path[:, t] = x
            x = Â * x
        end
        return x_path
    end

Let's observe these matrices for the baseline model

.. code-block:: julia

    lm = LakeModel()
    A, Â = transition_matrices(lm)
    A

.. code-block:: julia

    Â

And a revised model

.. code-block:: julia

    lm = LakeModel(α = 2.0)
    A, Â = transition_matrices(lm)
    A

.. code-block:: julia

    Â

.. code-block:: julia
    :class: test

    @testset begin
        @test lm.α ≈ 2.0
        @test A[1][1] ≈ 0.7235062600000001
    end

Aggregate Dynamics
------------------

Let's run a simulation under the default parameters (see above) starting from :math:`X_0 = (12, 138)`

.. code-block:: julia

    lm = LakeModel()
    N_0 = 150      # population
    e_0 = 0.92     # initial employment rate
    u_0 = 1 - e_0  # initial unemployment rate
    T = 50         # simulation length

    U_0 = u_0 * N_0
    E_0 = e_0 * N_0
    X_0 = [U_0; E_0]

    X_path = simulate_stock_path(lm, X_0, T)

    x1 = X_path[1, :]
    x2 = X_path[2, :]
    x3 = dropdims(sum(X_path, dims = 1), dims = 1)

    plt_unemp = plot(title = "Unemployment", 1:T, x1, color = :blue, lw = 2, grid = true, label = "")
    plt_emp = plot(title = "Employment", 1:T, x2, color = :blue, lw = 2, grid = true, label = "")
    plt_labor = plot(title = "Labor force", 1:T, x3, color = :blue, lw = 2, grid = true, label = "")

    plot(plt_unemp, plt_emp, plt_labor, layout = (3, 1), size = (800, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test x1[1] ≈ 11.999999999999995
        @test x2[2] ≈ 138.45447156
        @test x3[3] ≈ 151.25662086
    end

The aggregates :math:`E_t` and :math:`U_t` don't converge because  their sum :math:`E_t + U_t` grows at rate :math:`g`.

On the other hand, the vector of employment and unemployment rates :math:`x_t` can be in a steady state :math:`\bar x` if
there exists an :math:`\bar x`  such that

* :math:`\bar x = \hat A \bar x`

* the components satisfy :math:`\bar e + \bar u = 1`

This equation tells us that a steady state level :math:`\bar x` is an  eigenvector of :math:`\hat A` associated with a unit eigenvalue.

We also have :math:`x_t \to \bar x` as :math:`t \to \infty` provided that the remaining eigenvalue of :math:`\hat A` has modulus less that 1.

This is the case for our default parameters:

.. code-block:: julia

    lm = LakeModel()
    A, Â = transition_matrices(lm)
    e, f = eigvals(Â)
    abs(e), abs(f)

.. code-block:: julia
    :class: test

    @testset begin
        @test abs(e) ≈ 0.6953067378358462
        @test abs(f) ≈ 1.0
    end

Let's look at the convergence of the unemployment and employment rate to steady state levels (dashed red line)

.. code-block:: julia

    lm = LakeModel()
    e_0 = 0.92     # initial employment rate
    u_0 = 1 - e_0  # initial unemployment rate
    T = 50         # simulation length

    xbar = rate_steady_state(lm)
    x_0 = [u_0; e_0]
    x_path = simulate_rate_path(lm, x_0, T)

    plt_unemp = plot(title ="Unemployment rate", 1:T, x_path[1, :],color = :blue, lw = 2,
                     alpha = 0.5, grid = true, label = "")
    plot!(plt_unemp, [xbar[1]], color=:red, linetype = :hline, linestyle = :dash, lw = 2, label = "")
    plt_emp = plot(title = "Employment rate", 1:T, x_path[2, :],color = :blue, lw = 2, alpha = 0.5,
                   grid = true, label = "")
    plot!(plt_emp, [xbar[2]], color=:red, linetype = :hline, linestyle = :dash, lw = 2, label = "")
    plot(plt_unemp, plt_emp, layout = (2, 1), size=(700,500))

.. code-block:: julia
    :class: test

    @testset begin
        @test x_path[1,3] ≈ 0.08137725667264473
        @test x_path[2,7] ≈ 0.9176350068305223
    end

Dynamics of an Individual Worker
================================

An individual worker's employment dynamics are governed by a :doc:`finite state Markov process <../tools_and_techniques/finite_markov>`.

The worker can be in one of two states:

* :math:`s_t=0` means unemployed

* :math:`s_t=1` means employed

Let's start off under the assumption that :math:`b = d = 0`.

The associated transition matrix is then

.. math::

    P = \left(
            \begin{matrix}
                1 - \lambda & \lambda \\
                \alpha & 1 - \alpha
            \end{matrix}
        \right)

Let :math:`\psi_t` denote the :ref:`marginal distribution <mc_md>` over employment / unemployment states for the worker at time :math:`t`.

As usual, we regard it as a row vector.

We know :ref:`from an earlier discussion <mc_md>` that :math:`\psi_t` follows the law of motion

.. math::

    \psi_{t+1} = \psi_t P

We also know from the :doc:`lecture on finite Markov chains <../tools_and_techniques/finite_markov>`
that if :math:`\alpha \in (0, 1)` and :math:`\lambda \in (0, 1)`, then
:math:`P` has a unique stationary distribution, denoted here by :math:`\psi^*`.

The unique stationary distribution satisfies

.. math::

    \psi^*[0] = \frac{\alpha}{\alpha + \lambda}

Not surprisingly, probability mass on the unemployment state increases with
the dismissal rate and falls with the job finding rate rate.

Ergodicity
----------

Let's look at a typical lifetime of employment-unemployment spells.

We want to compute the average amounts of time an infinitely lived worker would spend employed and unemployed.

Let

.. math::

    \bar s_{u,T} := \frac1{T} \sum_{t=1}^T \mathbb 1\{s_t = 0\}

and

.. math::

    \bar s_{e,T} := \frac1{T} \sum_{t=1}^T \mathbb 1\{s_t = 1\}

(As usual, :math:`\mathbb 1\{Q\} = 1` if statement :math:`Q` is true and 0 otherwise)

These are the fraction of time a worker spends unemployed and employed, respectively, up until period :math:`T`.

If :math:`\alpha \in (0, 1)` and :math:`\lambda \in (0, 1)`, then :math:`P` is :ref:`ergodic <ergodicity>`, and hence we have

.. math::

    \lim_{T \to \infty} \bar s_{u, T} = \psi^*[0]
    \quad \text{and} \quad
    \lim_{T \to \infty} \bar s_{e, T} = \psi^*[1]

with probability one.

Inspection tells us that :math:`P` is exactly the transpose of :math:`\hat A` under the assumption :math:`b=d=0`.

Thus, the percentages of time that an  infinitely lived worker  spends employed and unemployed equal the fractions of workers employed and unemployed in the steady state distribution.

Convergence rate
----------------

How long does it take for time series sample averages to converge to cross sectional averages?

We can use `QuantEcon.jl's <http://quantecon.org/quantecon-jl>`__
`MarkovChain` type to investigate this.

Let's plot the path of the sample averages over 5,000 periods

.. code-block:: julia

    using QuantEcon, Roots, Random

.. code-block:: julia

    lm = LakeModel(d = 0, b = 0)
    T = 5000                        # Simulation length

    @unpack α, λ = lm
    P = [(1 - λ)     λ
         α      (1 - α)]

.. code-block:: julia

    Random.seed!(42)
    mc = MarkovChain(P, [0; 1])     # 0=unemployed, 1=employed
    xbar = rate_steady_state(lm)

    s_path = simulate(mc, T; init=2)
    s̄_e = cumsum(s_path) ./ (1:T)
    s̄_u = 1 .- s̄_e
    s_bars = [s̄_u s̄_e]

    plt_unemp = plot(title = "Percent of time unemployed", 1:T, s_bars[:,1],color = :blue, lw = 2,
                     alpha = 0.5, label = "", grid = true)
    plot!(plt_unemp, [xbar[1]], linetype = :hline, linestyle = :dash, color=:red, lw = 2, label = "")
    plt_emp = plot(title = "Percent of time employed", 1:T, s_bars[:,2],color = :blue, lw = 2,
                   alpha = 0.5, label = "", grid = true)
    plot!(plt_emp, [xbar[2]], linetype = :hline, linestyle = :dash, color=:red, lw = 2, label = "")
    plot(plt_unemp, plt_emp, layout = (2, 1), size=(700,500))

.. code-block:: julia
  :class: test

  @testset begin
      @test xbar[1] ≈ 0.04391891891891919
      @test s_bars[end,end] ≈ 0.957
  end

The stationary probabilities are given by the dashed red line.

In this case it takes much of the sample for these two objects to converge.

This is largely due to the high persistence in the Markov chain.

Endogenous Job Finding Rate
===========================

We now make the hiring rate endogenous.

The transition rate from unemployment to employment will be determined by the McCall search model :cite:`McCall1970`.

All details relevant to the following discussion can be found in :doc:`our treatment <../dynamic_programming/mccall_model>` of that model.

Reservation Wage
----------------

The most important thing to remember about the model is that optimal decisions
are characterized by a reservation wage :math:`\bar w`.

*  If the wage offer :math:`w` in hand is greater than or equal to :math:`\bar w`, then the worker accepts.

*  Otherwise, the worker rejects.

As we saw in :doc:`our discussion of the model <../dynamic_programming/mccall_model>`, the reservation wage depends on the wage offer distribution and the parameters

* :math:`\alpha`, the separation rate

* :math:`\beta`, the discount factor

* :math:`\gamma`, the offer arrival rate

* :math:`c`, unemployment compensation

Linking the McCall Search Model to the Lake Model
-------------------------------------------------

Suppose that  all workers inside a lake model behave according to the McCall search model.

The exogenous probability of leaving employment remains :math:`\alpha`.

But their optimal decision rules determine the probability :math:`\lambda` of leaving unemployment.

This is now

.. math::
    :label: lake_lamda

    \lambda
    = \gamma \mathbb P \{ w_t \geq \bar w\}
    = \gamma \sum_{w' \geq \bar w} p(w')

Fiscal Policy
-------------

We can use the McCall search version of the Lake Model  to find an optimal level of unemployment insurance.

We assume that  the government sets unemployment compensation :math:`c`.

The government imposes a lump sum tax :math:`\tau` sufficient to finance total unemployment payments.

To attain a balanced budget at a steady state, taxes, the steady state unemployment rate :math:`u`, and the unemployment compensation rate must satisfy

.. math::

    \tau = u c

The lump sum tax applies to everyone, including unemployed workers.

Thus, the post-tax income of an employed worker with wage :math:`w` is :math:`w - \tau`.

The post-tax income of an unemployed worker is :math:`c - \tau`.

For each specification :math:`(c, \tau)` of government policy, we can solve for the worker's optimal reservation wage.

This determines :math:`\lambda` via :eq:`lake_lamda` evaluated at post tax wages, which in turn determines a steady state unemployment rate :math:`u(c, \tau)`.

For a given level of unemployment benefit :math:`c`, we can solve for a tax that balances the budget in the steady state

.. math::

    \tau = u(c, \tau) c

To evaluate alternative government tax-unemployment compensation pairs, we require a welfare criterion.

We use a steady state welfare criterion

.. math::

    W := e \,  {\mathbb E} [V \, | \,  \text{employed}] + u \,  U

where the notation :math:`V` and :math:`U` is as defined in the :doc:`McCall search model lecture <../dynamic_programming/mccall_model>`.

The wage offer distribution will be a discretized version of the lognormal distribution :math:`LN(\log(20),1)`, as shown in the next figure

.. figure:: /_static/figures/lake_distribution_wages.png
    :width: 80%

We take a period to be a month.

We set :math:`b` and :math:`d` to match monthly `birth <http://www.cdc.gov/nchs/fastats/births.htm>`_ and `death rates <http://www.cdc.gov/nchs/fastats/deaths.htm>`_, respectively, in the U.S. population

* :math:`b = 0.0124`

* :math:`d = 0.00822`

Following :cite:`davis2006flow`, we set :math:`\alpha`, the hazard rate of leaving employment, to

* :math:`\alpha = 0.013`

Fiscal Policy Code
-----------------------

We will make use of (with some tweaks) the code we wrote in the :doc:`McCall model lecture <../dynamic_programming/mccall_model>`, embedded below for convenience.

.. code-block:: julia

    function solve_mccall_model(mcm; U_iv = 1.0, V_iv = ones(length(mcm.w)), tol = 1e-5,
                                iter = 2_000)
        @unpack α, β, σ, c, γ, w, E, u = mcm

        # necessary objects
        u_w = u.(w, σ)
        u_c = u(c, σ)

        # Bellman operator T. Fixed point is x* s.t. T(x*) = x*
        function T(x)
            V = x[1:end-1]
            U = x[end]
            [u_w + β * ((1 - α) * V .+ α * U); u_c + β * (1 - γ) * U + β * γ * E * max.(U, V)]
        end

        # value function iteration
        x_iv = [V_iv; U_iv] # initial x val
        xstar = fixedpoint(T, x_iv, iterations = iter, xtol = tol).zero
        V = xstar[1:end-1]
        U = xstar[end]

        # compute the reservation wage
        w_barindex = searchsortedfirst(V .- U, 0.0)
        if w_barindex >= length(w) # if this is true, you never want to accept
            w̄ = Inf
        else
            w̄ = w[w_barindex] # otherwise, return the number
        end

        # return a NamedTuple, so we can select values by name
        return (V = V, U = U, w̄ = w̄)
    end

And the McCall object

.. code-block:: julia

    # a default utility function
    u(c, σ) = c > 0 ? (c^(1 - σ) - 1) / (1 - σ) : -10e-6

    # model constructor
    McCallModel = @with_kw (α = 0.2,
                            β = 0.98, # discount rate
                            γ = 0.7,
                            c = 6.0, # unemployment compensation
                            σ = 2.0,
                            u = u, # utility function
                            w = range(10, 20, length = 60), # wage values
                            E = Expectation(BetaBinomial(59, 600, 400))) # distribution over wage values

Now let's compute and plot welfare, employment, unemployment, and tax revenue as a
function of the unemployment compensation rate

.. code-block:: julia

    # some global variables that will stay constant
    α = 0.013
    α_q = (1 - (1 - α)^3)
    b_param = 0.0124
    d_param = 0.00822
    β = 0.98
    γ = 1.0
    σ = 2.0

    # the default wage distribution: a discretized log normal
    log_wage_mean, wage_grid_size, max_wage = 20, 200, 170
    w_vec = range(1e-3, max_wage, length = wage_grid_size + 1)

    logw_dist = Normal(log(log_wage_mean), 1)
    cdf_logw = cdf.(logw_dist, log.(w_vec))
    pdf_logw = cdf_logw[2:end] - cdf_logw[1:end-1]

    p_vec = pdf_logw ./ sum(pdf_logw)
    w_vec = (w_vec[1:end-1] + w_vec[2:end]) / 2

    E = expectation(Categorical(p_vec)) # expectation object

    function compute_optimal_quantities(c, τ)
        mcm = McCallModel(α = α_q,
                          β = β,
                          γ = γ,
                          c = c - τ, # post-tax compensation
                          σ = σ,
                          w = w_vec .- τ, # post-tax wages
                          E = E) # expectation operator

        @unpack V, U, w̄ = solve_mccall_model(mcm)
        indicator = wage -> wage > w̄
        λ = γ * E * indicator.(w_vec .- τ)

        return w̄, λ, V, U
    end

    function compute_steady_state_quantities(c, τ)
        w̄, λ_param, V, U = compute_optimal_quantities(c, τ)

        # compute steady state employment and unemployment rates
        lm = LakeModel(λ = λ_param, α = α_q, b = b_param, d = d_param)
        x = rate_steady_state(lm)
        u_rate, e_rate = x

        # compute steady state welfare
        indicator(wage) = wage > w̄
        indicator(wage) = wage > w̄
        decisions = indicator.(w_vec .- τ)
        w = (E * (V .* decisions)) / (E * decisions)
        welfare = e_rate .* w + u_rate .* U

        return u_rate, e_rate, welfare
    end

    function find_balanced_budget_tax(c)
        function steady_state_budget(t)
            u_rate, e_rate, w = compute_steady_state_quantities(c, t)
            return t - u_rate * c
        end

        τ = find_zero(steady_state_budget, (0.0, 0.9c))
        return τ
    end

    # levels of unemployment insurance we wish to study
    Nc = 60
    c_vec = range(5, 140, length = Nc)

    tax_vec = zeros(Nc)
    unempl_vec = similar(tax_vec)
    empl_vec = similar(tax_vec)
    welfare_vec = similar(tax_vec)

    for i in 1:Nc
        t = find_balanced_budget_tax(c_vec[i])
        u_rate, e_rate, welfare = compute_steady_state_quantities(c_vec[i], t)
        tax_vec[i] = t
        unempl_vec[i] = u_rate
        empl_vec[i] = e_rate
        welfare_vec[i] = welfare
    end

    plt_unemp = plot(title = "Unemployment", c_vec, unempl_vec, color = :blue, lw = 2, alpha=0.7,
                     label = "",grid = true)
    plt_tax = plot(title = "Tax", c_vec, tax_vec, color = :blue, lw = 2, alpha=0.7, label = "",
                   grid = true)
    plt_emp = plot(title = "Employment", c_vec, empl_vec, color = :blue, lw = 2, alpha=0.7, label = "",
                   grid = true)
    plt_welf = plot(title = "Welfare", c_vec, welfare_vec, color = :blue, lw = 2, alpha=0.7, label = "",
                    grid = true)

    plot(plt_unemp, plt_emp, plt_tax, plt_welf, layout = (2,2), size = (800, 700))

.. code-block:: julia
    :class: test

    @testset begin
        @test tax_vec[40] ≈ 55.78515130142509 atol = 1e-7
        @test empl_vec[50] ≈ 0.2787840354254594 atol = 1e-7
        @test welfare_vec[17] ≈ 49.047098920686786 atol = 1e-6
    end

Welfare first increases and then decreases as unemployment benefits rise.

The level that maximizes steady state welfare is approximately 62.

Exercises
=========

Exercise 1
----------

Consider an economy with initial stock  of workers :math:`N_0 = 100` at the
steady state level of employment in the baseline parameterization

* :math:`\alpha = 0.013`

* :math:`\lambda = 0.283`

* :math:`b = 0.0124`

* :math:`d = 0.00822`

(The values for :math:`\alpha` and :math:`\lambda` follow :cite:`davis2006flow`)

Suppose that in response to new legislation the hiring rate reduces to :math:`\lambda = 0.2`.

Plot the transition dynamics of the unemployment and employment stocks for 50 periods.

Plot the transition dynamics for the rates.

How long does the economy take to converge to its new steady state?

What is the new steady state level of employment?

Exercise 2
----------

Consider an economy with initial stock  of workers :math:`N_0 = 100` at the
steady state level of employment in the baseline parameterization.

Suppose that for 20 periods the birth rate was temporarily high (:math:`b = 0.0025`) and then returned to its original level.

Plot the transition dynamics of the unemployment and employment stocks for 50 periods.

Plot the transition dynamics for the rates.

How long does the economy take to return to its original steady state?

Solutions
==========

Exercise 1
----------

We begin by constructing an object containing the default parameters and assigning the
steady state values to `x0`

.. code-block:: julia

    lm = LakeModel()
    x0 = rate_steady_state(lm)
    println("Initial Steady State: $x0")

Initialize the simulation values

.. code-block:: julia

    N0 = 100
    T = 50

New legislation changes :math:`\lambda` to :math:`0.2`

.. code-block:: julia

    lm = LakeModel(λ = 0.2)

.. code-block:: julia

    xbar = rate_steady_state(lm) # new steady state
    X_path = simulate_stock_path(lm, x0 * N0, T)
    x_path = simulate_rate_path(lm, x0, T)
    println("New Steady State: $xbar")

Now plot stocks

.. code-block:: julia

    x1 = X_path[1, :]
    x2 = X_path[2, :]
    x3 = dropdims(sum(X_path, dims = 1), dims = 1)

    plt_unemp = plot(title = "Unemployment", 1:T, x1, color = :blue, grid = true, label = "",
                     bg_inside = :lightgrey)
    plt_emp = plot(title = "Employment", 1:T, x2, color = :blue, grid = true, label = "",
                   bg_inside = :lightgrey)
    plt_labor = plot(title = "Labor force", 1:T, x3, color = :blue, grid = true, label = "",
                     bg_inside = :lightgrey)

    plot(plt_unemp, plt_emp, plt_labor, layout = (3, 1), size = (800, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test x1[1] ≈ 8.266626766923284
        @test x2[2] ≈ 91.43632870031433
        @test x3[3] ≈ 100.83774723999996
    end

And how the rates evolve

.. code-block:: julia

    plt_unemp = plot(title = "Unemployment rate", 1:T, x_path[1,:], color = :blue, grid = true,
                     label = "", bg_inside = :lightgrey)
    plot!(plt_unemp, [xbar[1]], linetype = :hline, linestyle = :dash, color =:red, label = "")

    plt_emp = plot(title = "Employment rate", 1:T, x_path[2,:], color = :blue, grid = true,
                   label = "", bg_inside = :lightgrey)
    plot!(plt_emp, [xbar[2]], linetype = :hline, linestyle = :dash, color =:red, label = "")

    plot(plt_unemp, plt_emp, layout = (2, 1), size = (800, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test x_path[1,3] ≈ 0.09471014989625384
        @test x_path[2,7] ≈ 0.8936171021324064
    end

We see that it takes 20 periods for the economy to converge to it's new
steady state levels.

Exercise 2
----------

This next exercise has the economy experiencing a boom in entrances to
the labor market and then later returning to the original levels.

For 20 periods the economy has a new entry rate into the labor market.

Let's start off at the baseline parameterization and record the steady
state

.. code-block:: julia

    lm = LakeModel()
    x0 = rate_steady_state(lm)

.. code-block:: julia
    :class: test

    @testset begin
        @test x0[1] ≈ 0.08266626766923285
    end

Here are the other parameters:

.. code-block:: julia

    b̂ = 0.003
    T̂ = 20

Let's increase :math:`b` to the new value and simulate for 20 periods

.. code-block:: julia

    lm = LakeModel(b=b̂)
    X_path1 = simulate_stock_path(lm, x0 * N0, T̂)   # simulate stocks
    x_path1 = simulate_rate_path(lm, x0, T̂)         # simulate rates

Now we reset :math:`b` to the original value and then, using the state
after 20 periods for the new initial conditions, we simulate for the
additional 30 periods

.. code-block:: julia

    lm = LakeModel(b = 0.0124)
    X_path2 = simulate_stock_path(lm, X_path1[:, end-1], T-T̂+1)    # simulate stocks
    x_path2 = simulate_rate_path(lm, x_path1[:, end-1], T-T̂+1)     # simulate rates

Finally we combine these two paths and plot

.. code-block:: julia

    x_path = hcat(x_path1, x_path2[:, 2:end])  # note [2:] to avoid doubling period 20
    X_path = hcat(X_path1, X_path2[:, 2:end])

.. code-block:: julia

    x1 = X_path[1,:]
    x2 = X_path[2,:]
    x3 = dropdims(sum(X_path, dims = 1), dims = 1)

    plt_unemp = plot(title = "Unemployment", 1:T, x1, color = :blue, lw = 2, alpha = 0.7,
                     grid = true, label = "", bg_inside = :lightgrey)
    plot!(plt_unemp, ylims = extrema(x1) .+ (-1, 1))

    plt_emp = plot(title = "Employment", 1:T, x2, color = :blue, lw = 2, alpha = 0.7, grid = true,
                   label = "", bg_inside = :lightgrey)
    plot!(plt_emp, ylims = extrema(x2) .+ (-1, 1))

    plt_labor = plot(title = "Labor force", 1:T, x3, color = :blue, alpha = 0.7, grid = true,
                     label = "", bg_inside = :lightgrey)
    plot!(plt_labor, ylims = extrema(x3) .+ (-1, 1))
    plot(plt_unemp, plt_emp, plt_labor, layout = (3, 1), size = (800, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test x1[1] ≈ 8.266626766923284
        @test x2[2] ≈ 92.11681873319097
        @test x3[3] ≈ 98.95872483999996
    end

And the rates

.. code-block:: julia

    plt_unemp = plot(title = "Unemployment Rate", 1:T, x_path[1,:], color = :blue, grid = true,
                     label = "", bg_inside = :lightgrey, lw = 2)
    plot!(plt_unemp, [x0[1]], linetype = :hline, linestyle = :dash, color =:red, label = "", lw = 2)

    plt_emp = plot(title = "Employment Rate", 1:T, x_path[2,:], color = :blue, grid = true,
                   label = "", bg_inside = :lightgrey, lw = 2)
    plot!(plt_emp, [x0[2]], linetype = :hline, linestyle = :dash, color =:red, label = "", lw = 2)

    plot(plt_unemp, plt_emp, layout = (2, 1), size = (800, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test x_path[1,3] ≈ 0.06791408368459205
        @test x_path[2,7] ≈ 0.9429334437639298
    end
