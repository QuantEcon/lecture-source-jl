.. _lqc:

.. include:: /_static/includes/header.raw

.. highlight:: julia

*******************************
LQ Dynamic Programming Problems
*******************************

.. index::
    single: LQ Control

.. contents:: :depth: 2

Overview
========

Linear quadratic (LQ) control refers to a class of dynamic optimization problems that have found applications in almost every scientific field.

This lecture provides an introduction to LQ control and its economic applications.

As we will see, LQ systems have a simple structure that makes them an excellent workhorse for a wide variety of economic problems.

Moreover, while the linear-quadratic structure is restrictive, it is in fact far more flexible than it may appear initially.

These themes appear repeatedly below.

Mathematically, LQ control problems are closely related to :doc:`the Kalman filter <../tools_and_techniques/kalman>`.

* Recursive formulations of linear-quadratic control problems and Kalman filtering problems both involve matrix **Riccati equations**.

* Classical formulations of linear control and linear filtering problems make use of similar matrix decompositions (see for example :doc:`this lecture <../time_series_models/lu_tricks>` and :doc:`this lecture <../time_series_models/classical_filtering>`).

In reading what follows, it will be useful to have some familiarity with

* matrix manipulations

* vectors of random variables

* dynamic programming and the Bellman equation (see for example :doc:`this lecture <../dynamic_programming/short_path>` and :doc:`this lecture <../dynamic_programming/optgrowth>`)

For additional reading on LQ control, see, for example,

* :cite:`Ljungqvist2012`, chapter 5

* :cite:`HansenSargent2008`, chapter 4

* :cite:`HernandezLermaLasserre1996`, section 3.5

In order to focus on computation, we leave longer proofs to these sources (while trying to provide as much intuition as possible).

Introduction
============

The "linear" part of LQ is a linear law of motion for the state, while the "quadratic" part refers to preferences.

Let's begin with the former, move on to the latter, and then put them together into an optimization problem.

The Law of Motion
-----------------

Let :math:`x_t` be a vector describing the state of some economic system.

Suppose that :math:`x_t` follows a linear law of motion given by

.. math::
    :label: lq_lom

    x_{t+1} = A x_t + B u_t + C w_{t+1},
    \qquad t = 0, 1, 2, \ldots

Here

* :math:`u_t` is a "control" vector, incorporating choices available to a decision maker confronting the current state :math:`x_t`.

* :math:`\{w_t\}` is an uncorrelated zero mean shock process satisfying :math:`\mathbb E w_t w_t' = I`, where the right-hand side is the identity matrix.

Regarding the dimensions

* :math:`x_t` is :math:`n \times 1`, :math:`A` is :math:`n \times n`

* :math:`u_t` is :math:`k \times 1`, :math:`B` is :math:`n \times k`

* :math:`w_t` is :math:`j \times 1`, :math:`C` is :math:`n \times j`

Example 1
^^^^^^^^^

Consider a household budget constraint given by

.. math::

    a_{t+1} + c_t = (1 + r) a_t + y_t

Here :math:`a_t` is assets, :math:`r` is a fixed interest rate, :math:`c_t` is
current consumption, and :math:`y_t` is current non-financial income.

If we suppose that :math:`\{ y_t \}` is serially uncorrelated and :math:`N(0,
\sigma^2)`, then, taking :math:`\{ w_t \}` to be standard normal, we can write
the system as

.. math::

    a_{t+1} = (1 + r) a_t - c_t + \sigma w_{t+1}

This is clearly a special case of :eq:`lq_lom`, with assets being the state and consumption being the control.

.. _lq_hhp:

Example 2
^^^^^^^^^

One unrealistic feature of the previous model is that non-financial income has a zero mean and is often negative.

This can easily be overcome by adding a sufficiently large mean.

Hence in this example we take :math:`y_t = \sigma w_{t+1} + \mu` for some positive real number :math:`\mu`.

Another alteration that's useful to introduce (we'll see why soon) is to
change the control variable from consumption
to the deviation of consumption from some "ideal" quantity :math:`\bar c`.

(Most parameterizations will be such that :math:`\bar c` is large relative to the amount of consumption that is attainable in each period, and hence the household wants to increase consumption)

For this reason, we now take our control to be :math:`u_t := c_t - \bar c`.

In terms of these variables, the budget constraint :math:`a_{t+1} = (1 + r) a_t - c_t + y_t` becomes

.. math::
    :label: lq_lomwc

    a_{t+1} = (1 + r) a_t - u_t - \bar c + \sigma w_{t+1} + \mu

How can we write this new system in the form of equation :eq:`lq_lom`?

If, as in the previous example, we take :math:`a_t` as the state, then we run into a problem:
the law of motion contains some constant terms on the right-hand side.

This means that we are dealing with an *affine* function, not a linear one
(recall :ref:`this discussion <la_linear_map>`).

Fortunately, we can easily circumvent this problem by adding an extra state variable.

In particular, if we write

.. math::
    :label: lq_lowmc

    \left(
    \begin{array}{c}
    a_{t+1} \\
    1
    \end{array}
    \right) =
    \left(
    \begin{array}{cc}
    1 + r & -\bar c + \mu \\
    0     & 1
    \end{array}
    \right)
    \left(
    \begin{array}{c}
    a_t \\
    1
    \end{array}
    \right) +
    \left(
    \begin{array}{c}    -1 \\
    0
    \end{array}
    \right)
    u_t +
    \left(
    \begin{array}{c}
    \sigma \\
    0
    \end{array}
    \right)
    w_{t+1}

then the first row is equivalent to :eq:`lq_lomwc`.

Moreover, the model is now linear, and can be written in the form of
:eq:`lq_lom` by setting

.. math::
    :label: lq_lowmc2

    x_t :=
    \left(
    \begin{array}{c}
    a_t \\
    1
    \end{array}
    \right),
    \quad
    A :=
    \left(
    \begin{array}{cc}
    1 + r & -\bar c + \mu \\
    0     & 1
    \end{array}
    \right),
    \quad
    B :=
    \left(
    \begin{array}{c}
    -1 \\
    0
    \end{array}
    \right),
    \quad
    C :=
    \left(
    \begin{array}{c}
    \sigma \\
    0
    \end{array}
    \right)

In effect, we've bought ourselves linearity by adding another state.

Preferences
-----------

In the LQ model, the aim is to minimize a flow of losses, where time-:math:`t` loss is given by the quadratic expression

.. math::
    :label: lq_pref_flow

    x_t' R x_t + u_t' Q u_t

Here

* :math:`R` is assumed to be :math:`n \times n`, symmetric and nonnegative definite

* :math:`Q` is assumed to be :math:`k \times k`, symmetric and positive definite

.. note::
    In fact, for many economic problems, the definiteness conditions on :math:`R` and :math:`Q` can be relaxed.  It is sufficient that certain submatrices of :math:`R` and :math:`Q` be nonnegative definite. See :cite:`HansenSargent2008` for details

Example 1
^^^^^^^^^

A very simple example that satisfies these assumptions is to take :math:`R`
and :math:`Q` to be identity matrices, so that current loss is

.. math::

    x_t' I x_t + u_t' I u_t = \| x_t \|^2 + \| u_t \|^2

Thus, for both the state and the control, loss is measured as squared distance from the origin.

(In fact the general case :eq:`lq_pref_flow` can also be understood in this
way, but with :math:`R` and :math:`Q` identifying other -- non-Euclidean -- notions of "distance" from the zero vector).

Intuitively, we can often think of the state :math:`x_t` as representing deviation from a target, such
as

* deviation of inflation from some target level

* deviation of a firm's capital stock from some desired quantity

The aim is to put the state close to the target, while using  controls parsimoniously.

Example 2
^^^^^^^^^

In the household problem :ref:`studied above <lq_hhp>`, setting :math:`R=0`
and :math:`Q=1` yields preferences

.. math::

    x_t' R x_t + u_t' Q u_t = u_t^2 = (c_t - \bar c)^2

Under this specification, the household's current loss is the squared deviation of consumption from the ideal level :math:`\bar c`.

Optimality -- Finite Horizon
============================

.. index::
    single: LQ Control; Optimality (Finite Horizon)

Let's now be precise about the optimization problem we wish to consider, and look at how to solve it.

The Objective
-------------

We will begin with the finite horizon case, with terminal time :math:`T \in \mathbb N`.

In this case, the aim is to choose a sequence of controls :math:`\{u_0, \ldots, u_{T-1}\}` to minimize the objective

.. math::
    :label: lq_object

    \mathbb E \,
    \left\{
        \sum_{t=0}^{T-1} \beta^t (x_t' R x_t + u_t' Q u_t) + \beta^T x_T' R_f x_T
    \right\}

subject to the law of motion :eq:`lq_lom` and initial state :math:`x_0`.

The new objects introduced here are :math:`\beta` and the matrix :math:`R_f`.

The scalar :math:`\beta` is the discount factor, while :math:`x' R_f x` gives terminal loss associated with state :math:`x`.

Comments:

* We assume :math:`R_f` to be :math:`n \times n`, symmetric and nonnegative definite

* We allow :math:`\beta = 1`, and hence include the undiscounted case

* :math:`x_0` may itself be random, in which case we require it to be independent of the shock sequence :math:`w_1, \ldots, w_T`

.. _lq_cp:

Information
-----------

There's one constraint we've neglected to mention so far, which is that the
decision maker who solves this LQ problem knows only the present and the past,
not the future.

To clarify this point, consider the sequence of controls :math:`\{u_0, \ldots, u_{T-1}\}`.

When choosing these controls, the decision maker is permitted to take into account the effects of the shocks
:math:`\{w_1, \ldots, w_T\}` on the system.

However, it is typically assumed --- and will be assumed here --- that the
time-:math:`t` control :math:`u_t` can  be made with knowledge of past and
present shocks only.

The fancy `measure-theoretic <https://en.wikipedia.org/wiki/Measure_%28mathematics%29>`_ way of saying this is that :math:`u_t` must be measurable with respect to the :math:`\sigma`-algebra generated by :math:`x_0, w_1, w_2,
\ldots, w_t`.

This is in fact equivalent to stating that :math:`u_t` can be written in the form :math:`u_t = g_t(x_0, w_1, w_2, \ldots, w_t)` for some Borel measurable function :math:`g_t`.

(Just about every function that's useful for applications is Borel measurable,
so, for the purposes of intuition, you can read that last phrase as "for some function :math:`g_t`").

Now note that :math:`x_t` will ultimately depend on the realizations of :math:`x_0, w_1, w_2, \ldots, w_t`.

In fact it turns out that :math:`x_t` summarizes all the information about  these historical  shocks that the decision maker needs to set controls optimally.

More precisely, it can be shown that any optimal control :math:`u_t` can always be written as a function of the current state alone.

Hence in what follows we restrict attention to control policies (i.e., functions) of the form :math:`u_t = g_t(x_t)`.

Actually, the preceding discussion applies to all standard dynamic programming problems.

What's special about the LQ case is that -- as we shall soon see ---  the optimal :math:`u_t` turns out to be a linear function of :math:`x_t`.

Solution
--------

To solve the finite horizon LQ problem we can use a dynamic programming
strategy based on backwards induction that is conceptually similar to the approach adopted in :doc:`this lecture <../dynamic_programming/short_path>`.

For reasons that will soon become clear, we first introduce the notation :math:`J_T(x) = x' R_f x`.

Now consider the problem of the decision maker in the second to last period.

In particular, let the time be :math:`T-1`, and suppose that the
state is :math:`x_{T-1}`.

The decision maker must trade off current and (discounted) final losses, and hence
solves

.. math::

    \min_u \{
    x_{T-1}' R x_{T-1} + u' Q u + \beta \,
    \mathbb E J_T(A x_{T-1} + B u + C w_T)
    \}

At this stage, it is convenient to define the function

.. math::
    :label: lq_lsm

    J_{T-1} (x) =
    \min_u \{
    x' R x + u' Q u + \beta \,
    \mathbb E J_T(A x + B u + C w_T)
    \}

The function :math:`J_{T-1}` will be called the :math:`T-1` value function, and :math:`J_{T-1}(x)` can be thought of as representing total "loss-to-go" from state :math:`x` at time :math:`T-1` when the decision maker behaves optimally.

Now let's step back to :math:`T-2`.

For a decision maker at :math:`T-2`, the value :math:`J_{T-1}(x)` plays a role analogous to that played by the terminal loss :math:`J_T(x) = x' R_f x` for the decision maker at :math:`T-1`.

That is, :math:`J_{T-1}(x)` summarizes the future loss associated with moving to state :math:`x`.

The decision maker chooses her control :math:`u` to trade off current loss against future loss, where

* the next period state is :math:`x_{T-1} = Ax_{T-2} + B u + C w_{T-1}`, and hence depends on the choice of current control

* the "cost" of landing in state :math:`x_{T-1}` is :math:`J_{T-1}(x_{T-1})`

Her problem is therefore

.. math::

    \min_u
    \{
    x_{T-2}' R x_{T-2} + u' Q u + \beta \,
    \mathbb E J_{T-1}(Ax_{T-2} + B u + C w_{T-1})
    \}

Letting

.. math::

    J_{T-2} (x)
    = \min_u
    \{
    x' R x + u' Q u + \beta \,
    \mathbb E J_{T-1}(Ax + B u + C w_{T-1})
    \}

the pattern for backwards induction is now clear.

In particular, we define a sequence of value functions :math:`\{J_0, \ldots, J_T\}` via

.. math::

    J_{t-1} (x)
    = \min_u
    \{
    x' R x + u' Q u + \beta \,
    \mathbb E J_{t}(Ax + B u + C w_t)
    \}
    \quad \text{and} \quad
    J_T(x) = x' R_f x

The first equality is the Bellman equation from dynamic programming theory specialized to the finite horizon LQ problem.

Now that we have :math:`\{J_0, \ldots, J_T\}`, we can obtain the optimal controls.

As a first step, let's find out what the value functions look like.

It turns out that every :math:`J_t` has the form :math:`J_t(x) = x' P_t x + d_t` where :math:`P_t` is a :math:`n \times n` matrix and :math:`d_t` is a constant.

We can show this by induction, starting from :math:`P_T := R_f` and :math:`d_T = 0`.

Using this notation, :eq:`lq_lsm` becomes

.. math::
    :label: lq_fswb

    J_{T-1} (x) =
    \min_u \{
    x' R x + u' Q u + \beta \,
    \mathbb E (A x + B u + C w_T)' P_T (A x + B u + C w_T)
    \}

To obtain the minimizer, we can take the derivative of the r.h.s. with respect to :math:`u` and set it equal to zero.

Applying the relevant rules of :ref:`matrix calculus <la_mcalc>`, this gives

.. math::
    :label: lq_oc0

    u  = - (Q + \beta B' P_T B)^{-1} \beta B' P_T A x

Plugging this back into :eq:`lq_fswb` and rearranging yields

.. math::

    J_{T-1} (x) = x' P_{T-1} x + d_{T-1}

where

.. math::
    :label: lq_finr

    P_{T-1}
    = R - \beta^2 A' P_T B (Q + \beta B' P_T B)^{-1} B' P_T A +
    \beta A' P_T A

and

.. math::
    :label: lq_finrd

    d_{T-1} := \beta \mathop{\mathrm{trace}}(C' P_T C)

(The algebra is a good exercise --- we'll leave it up to you).

If we continue working backwards in this manner, it soon becomes clear that :math:`J_t (x) = x' P_t x + d_t` as claimed, where :math:`\{P_t\}` and :math:`\{d_t\}` satisfy the recursions

.. math::
    :label: lq_pr

    P_{t-1}
    = R - \beta^2 A' P_t B (Q + \beta B' P_t B)^{-1} B' P_t A +
    \beta A' P_t A
    \quad \text{with } \quad
    P_T = R_f

and

.. math::
    :label: lq_dd

    d_{t-1} = \beta (d_t + \mathop{\mathrm{trace}}(C' P_t C))
    \quad \text{with } \quad
    d_T = 0

Recalling :eq:`lq_oc0`, the minimizers from these backward steps are

.. math::
    :label: lq_oc

    u_t  = - F_t x_t
    \quad \text{where} \quad
    F_t := (Q + \beta B' P_{t+1} B)^{-1} \beta B' P_{t+1} A

These are the linear optimal control policies we :ref:`discussed above <lq_cp>`.

In particular,  the sequence of controls given by :eq:`lq_oc` and :eq:`lq_lom` solves our finite horizon LQ problem.

Rephrasing this more precisely, the sequence :math:`u_0, \ldots, u_{T-1}` given by

.. math::
    :label: lq_xud

    u_t = - F_t x_t
    \quad \text{with} \quad
    x_{t+1} = (A - BF_t) x_t + C w_{t+1}

for :math:`t = 0, \ldots, T-1` attains the minimum of :eq:`lq_object` subject to our constraints.

Implementation
==============

We will use code from `lqcontrol.jl <https://github.com/QuantEcon/QuantEcon.jl/blob/master/src/lqcontrol.jl>`__
in `QuantEcon.jl <http://quantecon.org/julia_index.html>`_
to solve finite and infinite horizon linear quadratic control problems.

In the module, the various updating, simulation and fixed point methods  act on a type  called ``LQ``, which includes

* Instance data:

    * The required parameters :math:`Q, R, A, B` and optional parameters `C, β, T, R_f, N` specifying a given LQ model

        * set :math:`T` and :math:`R_f` to ``None`` in the infinite horizon case

        * set ``C = None`` (or zero) in the deterministic case

    * the value function and policy data

        * :math:`d_t, P_t, F_t` in the finite horizon case

        * :math:`d, P, F` in the infinite horizon case

* Methods:

    * ``update_values`` --- shifts :math:`d_t, P_t, F_t` to their :math:`t-1` values via :eq:`lq_pr`, :eq:`lq_dd` and :eq:`lq_oc`

    * ``stationary_values`` --- computes :math:`P, d, F` in the infinite horizon case

    * ``compute_sequence`` ---- simulates the dynamics of :math:`x_t, u_t, w_t` given :math:`x_0` and assuming standard normal shocks

.. _lq_mfpa:

An Application
--------------

Early Keynesian models assumed that households have a constant marginal
propensity to consume from current income.

Data contradicted the constancy of the marginal propensity to consume.

In response, Milton Friedman, Franco Modigliani and others built models
based on a consumer's preference for an intertemporally smooth consumption stream.

(See, for example, :cite:`Friedman1956` or :cite:`ModiglianiBrumberg1954`).

One property of those models is that households purchase and sell financial assets to make consumption streams smoother than income streams.

The household savings problem :ref:`outlined above <lq_hhp>` captures these ideas.

The optimization problem for the household is to choose a consumption sequence in order to minimize

.. math::
    :label: lq_pio

    \mathbb E \,
    \left\{
        \sum_{t=0}^{T-1} \beta^t (c_t - \bar c)^2 + \beta^T q a_T^2
    \right\}

subject to the sequence of budget constraints :math:`a_{t+1} = (1 + r) a_t - c_t + y_t, \ t \geq 0`.

Here :math:`q` is a large positive constant, the role of which is to induce the consumer to target zero debt at the end of her life.

(Without such a constraint, the optimal choice is to choose :math:`c_t = \bar c` in each period, letting assets adjust accordingly).

As before we set :math:`y_t = \sigma w_{t+1} + \mu` and :math:`u_t := c_t - \bar c`, after which the constraint can be written as in :eq:`lq_lomwc`.

We saw how this constraint could be manipulated into the LQ formulation :math:`x_{t+1} =
Ax_t + Bu_t + Cw_{t+1}` by setting :math:`x_t = (a_t \; 1)'` and using the definitions in :eq:`lq_lowmc2`.

To match with this state and control, the objective function :eq:`lq_pio` can
be written in the form of :eq:`lq_object` by choosing

.. math::

    Q := 1,
    \quad
    R :=
    \left(
    \begin{array}{cc}
    0 & 0 \\
    0 & 0
    \end{array}
    \right),
    \quad \text{and} \quad
    R_f :=
    \left(
    \begin{array}{cc}
    q & 0 \\
    0 & 0
    \end{array}
    \right)

Now that the problem is expressed in LQ form, we can proceed to the solution
by applying :eq:`lq_pr` and :eq:`lq_oc`.

After generating shocks :math:`w_1, \ldots, w_T`, the dynamics for assets and
consumption can be simulated via :eq:`lq_xud`.

The following figure was computed using :math:`r = 0.05, \beta = 1 / (1 + r), \bar c = 2,  \mu = 1, \sigma = 0.25, T = 45` and :math:`q = 10^6`.

The shocks :math:`\{w_t\}` were taken to be iid and standard normal.

Setup
-----

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia
    :class: test

    using Test, Random

.. code-block:: julia

    using LinearAlgebra, Statistics
    using Plots, Plots.PlotMeasures, QuantEcon

.. code-block:: julia
    :class: test

    Random.seed!(42);

.. code-block:: julia

    # model parameters
    r = 0.05
    β = 1 / (1 + r)
    T = 45
    c̄ = 2.0
    σ = 0.25
    μ = 1.0
    q = 1e6

    # formulate as an LQ problem
    Q = 1.0
    R = zeros(2, 2)
    Rf = zeros(2, 2); Rf[1, 1] = q
    A = [1 + r -c̄ + μ; 0  1]
    B = [-1.0, 0]
    C = [σ, 0]

    # compute solutions and simulate
    lq = QuantEcon.LQ(Q, R, A, B, C; bet = β, capT = T, rf = Rf)
    x0 = [0.0, 1]
    xp, up, wp = compute_sequence(lq, x0)

    # convert back to assets, consumption and income
    assets = vec(xp[1, :]) # a_t
    c = vec(up .+ c̄) # c_t
    income = vec(σ * wp[1, 2:end] .+ μ) # y_t

    # plot results
    p = plot([assets, c, zeros(T + 1), income, cumsum(income .- μ)],
             lab = ["assets" "consumption" "" "non-financial income" "cumulative unanticipated income"],
             color = [:blue :green :black :orange :red],
             xaxis = "Time", layout = (2, 1),
             bottom_margin = 20mm, size = (600, 600))

.. code-block:: julia
    :class: test

    @testset "First Plots Tests" begin
        @test income[3] ≈ 0.9812822443525681 # test determinism and intermediate calculations
        @test up[4] ≈ -1.010200105783321 # test downstream invariance
    end

The top panel shows the time path of consumption :math:`c_t` and income :math:`y_t` in the simulation.

As anticipated by the discussion on consumption smoothing, the time path of
consumption is much smoother than that for income.

(But note that  consumption becomes more irregular towards the end of life,
when the zero final asset requirement impinges more on consumption choices).

The second panel in the figure shows that the time path of assets :math:`a_t` is
closely correlated with cumulative unanticipated income, where the latter is defined as

.. math::

    z_t := \sum_{j=0}^t \sigma w_t

A key message is that unanticipated windfall gains are saved rather
than consumed, while unanticipated negative shocks are met by reducing assets.

(Again, this relationship breaks down towards the end of life due to the zero final asset requirement).

These results are relatively robust to changes in parameters.

For example, let's increase :math:`\beta` from :math:`1 / (1 + r) \approx 0.952` to :math:`0.96` while keeping other parameters fixed.

This consumer is slightly more patient than the last one, and hence puts
relatively more weight on later consumption values

.. code-block:: julia
    :class: test

    Random.seed!(42);

.. code-block:: julia
    :class: collapse

    # compute solutions and simulate
    lq = QuantEcon.LQ(Q, R, A, B, C; bet = 0.96, capT = T, rf = Rf)
    x0 = [0.0, 1]
    xp, up, wp = compute_sequence(lq, x0)

    # convert back to assets, consumption and income
    assets = vec(xp[1, :]) # a_t
    c = vec(up .+ c̄) # c_t
    income = vec(σ * wp[1, 2:end] .+ μ) # y_t

    # plot results
    p = plot([assets, c, zeros(T + 1), income, cumsum(income .- μ)],
             lab = ["assets" "consumption" "" "non-financial income" "cumulative unanticipated income"],
             color = [:blue :green :black :orange :red],
             xaxis = "Time", layout = (2, 1),
             bottom_margin = 20mm, size = (600, 600))

We now have a slowly rising consumption stream and a hump-shaped build
up of assets in the middle periods to fund rising consumption.

However, the essential features are the same: consumption is smooth relative to income, and assets are strongly positively correlated with cumulative unanticipated income.

Extensions and Comments
=======================

Let's now consider a number of standard extensions to the LQ problem treated above.

Time-Varying Parameters
-----------------------

In some settings it can be desirable to allow :math:`A, B, C, R` and :math:`Q` to depend on :math:`t`.

For the sake of simplicity, we've chosen not to treat this extension in our implementation given below.

However, the loss of generality is not as large as you might first imagine.

In fact, we can tackle many models with time-varying parameters by suitable choice of state variables.

One illustration is given :ref:`below <lq_nsi>`.

For further examples and a more systematic treatment, see :cite:`HansenSargent2013`, section 2.4.

.. _lq_cpt:

Adding a Cross-Product Term
---------------------------

In some LQ problems, preferences include a cross-product term :math:`u_t' N x_t`, so that the objective function becomes

.. math::
    :label: lq_object_cp

    \mathbb E \,
    \left\{
        \sum_{t=0}^{T-1} \beta^t (x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t) + \beta^T x_T' R_f x_T
    \right\}

Our results extend to this case in a straightforward way.

The sequence :math:`\{P_t\}` from :eq:`lq_pr` becomes

.. math::
    :label: lq_pr_cp

    P_{t-1}
    = R - (\beta B' P_t A + N)' (Q + \beta B' P_t B)^{-1} (\beta B' P_t A + N) +
    \beta A' P_t A
    \quad \text{with } \quad
    P_T = R_f

The policies in :eq:`lq_oc` are modified to

.. math::
    :label: lq_oc_cp

    u_t  = - F_t x_t
    \quad \text{where} \quad
    F_t := (Q + \beta B' P_{t+1} B)^{-1} (\beta B' P_{t+1} A + N)

The sequence :math:`\{d_t\}` is unchanged from :eq:`lq_dd`.

We leave interested readers to confirm these results (the calculations are long but not overly difficult).

.. _lq_ih:

Infinite Horizon
----------------

.. index::
    single: LQ Control; Infinite Horizon

Finally, we consider the infinite horizon case, with :ref:`cross-product term <lq_cpt>`, unchanged dynamics and
objective function given by

.. math::
    :label: lq_object_ih

    \mathbb E \,
    \left\{
        \sum_{t=0}^{\infty} \beta^t (x_t' R x_t + u_t' Q u_t + 2 u_t' N x_t)
    \right\}

In the infinite horizon case, optimal policies can depend on time
only if time itself is a component of the  state vector :math:`x_t`.

In other words, there exists a fixed matrix :math:`F` such that :math:`u_t = -
F x_t` for all :math:`t`.

That decision rules are constant over time is intuitive --- after all, the decision maker faces the
same infinite horizon at every stage, with only the current state changing.

Not surprisingly, :math:`P` and :math:`d` are also constant.

The stationary matrix :math:`P` is the solution to the
`discrete time algebraic Riccati equation <https://en.wikipedia.org/wiki/Algebraic_Riccati_equation>`_.

.. _riccati_equation:

.. math::
    :label: lq_pr_ih

    P = R - (\beta B' P A + N)' (Q + \beta B' P B)^{-1} (\beta B' P A + N) +
    \beta A' P A

Equation :eq:`lq_pr_ih` is also called the *LQ Bellman equation*, and the map
that sends a given :math:`P` into the right-hand side of :eq:`lq_pr_ih` is
called the *LQ Bellman operator*.

The stationary optimal policy for this model is

.. math::
    :label: lq_oc_ih

    u  = - F x
    \quad \text{where} \quad
    F = (Q + \beta B' P B)^{-1} (\beta B' P A + N)

The sequence :math:`\{d_t\}` from :eq:`lq_dd` is replaced by the constant value

.. math::
    :label: lq_dd_ih

    d
    := \mathop{\mathrm{trace}}(C' P C) \frac{\beta}{1 - \beta}

The state evolves according to the time-homogeneous process :math:`x_{t+1} = (A - BF) x_t + C w_{t+1}`.

An example infinite horizon problem is treated :ref:`below <lqc_mwac>`.

.. _lq_cert_eq:

Certainty Equivalence
---------------------

Linear quadratic control problems of the class discussed above have the property of *certainty equivalence*.

By this we mean that the optimal policy :math:`F` is not affected by the parameters in :math:`C`, which specify the shock process.

This can be confirmed by inspecting :eq:`lq_oc_ih` or :eq:`lq_oc_cp`.

It follows that we can ignore uncertainty when solving for optimal behavior, and plug it back in when examining optimal state dynamics.

Further Applications
====================

.. _lq_nsi:

Application 1: Age-Dependent Income Process
-------------------------------------------

:ref:`Previously <lq_mfpa>` we studied a permanent income model that generated consumption smoothing.

One unrealistic feature of that model is the assumption that the mean of the random income process does not depend on the consumer's age.

A more realistic income profile is one that rises in early working life, peaks towards the middle and maybe declines toward end of working life, and falls more during retirement.

In this section, we will model this rise and fall as a symmetric inverted "U" using a polynomial in age.

As before, the consumer seeks to minimize

.. math::
    :label: lq_pip

    \mathbb E \,
    \left\{
        \sum_{t=0}^{T-1} \beta^t (c_t - \bar c)^2 + \beta^T q a_T^2
    \right\}

subject to :math:`a_{t+1} = (1 + r) a_t - c_t + y_t, \ t \geq 0`.

For income we now take :math:`y_t = p(t) + \sigma w_{t+1}` where :math:`p(t) := m_0 + m_1 t + m_2 t^2`.

(In :ref:`the next section <lq_nsi2>` we employ some tricks to implement a more sophisticated model).

The coefficients :math:`m_0, m_1, m_2` are chosen such that :math:`p(0)=0, p(T/2) = \mu,` and :math:`p(T)=0`.

You can confirm that the specification :math:`m_0 = 0, m_1 = T \mu / (T/2)^2, m_2 = - \mu / (T/2)^2` satisfies these constraints.

To put this into an LQ setting, consider the budget constraint, which becomes

.. math::
    :label: lq_hib

    a_{t+1} = (1 + r) a_t - u_t - \bar c + m_1 t + m_2 t^2 + \sigma w_{t+1}

The fact that :math:`a_{t+1}` is a linear function of
:math:`(a_t, 1, t, t^2)` suggests taking these four variables as the state
vector :math:`x_t`.

Once a good choice of state and control (recall :math:`u_t = c_t - \bar c`)
has been made, the remaining specifications fall into place relatively easily.

Thus, for the dynamics we set

.. math::
    :label: lq_lowmc3

    x_t :=
    \left(
    \begin{array}{c}
    a_t \\
    1 \\
    t \\
    t^2
    \end{array}
    \right),
    \quad
    A :=
    \left(
    \begin{array}{cccc}
    1 + r & -\bar c & m_1 & m_2 \\
    0     & 1       & 0   & 0   \\
    0     & 1       & 1   & 0   \\
    0     & 1       & 2   & 1   \\
    \end{array}
    \right),
    \quad
    B :=
    \left(
    \begin{array}{c}
    -1 \\
    0 \\
    0 \\
    0
    \end{array}
    \right),
    \quad
    C :=
    \left(
    \begin{array}{c}
    \sigma \\
    0 \\
    0 \\
    0
    \end{array}
    \right)

If you expand the expression :math:`x_{t+1} = A x_t + B u_t + C w_{t+1}` using
this specification, you will find that assets follow :eq:`lq_hib` as desired,
and that the other state variables also update appropriately.

To implement preference specification :eq:`lq_pip` we take

.. math::
    :label: lq_4sp

    Q := 1,
    \quad
    R :=
    \left(
    \begin{array}{cccc}
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
    \end{array}
    \right)
    \quad \text{and} \quad
    R_f :=
    \left(
    \begin{array}{cccc}
    q & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
    \end{array}
    \right)

The next figure shows a simulation of consumption and assets computed using
the ``compute_sequence`` method of ``lqcontrol.jl`` with initial assets set to zero.

.. _solution_lqc_ex1_fig:

.. figure:: /_static/figures/solution_lqc_ex1.png

Once again, smooth consumption is a dominant feature of the sample  paths.

The asset path exhibits dynamics consistent with standard life cycle theory.

Exercise 1 gives the full set of parameters used here and asks you to replicate the figure.

.. _lq_nsi2:

Application 2: A Permanent Income Model with Retirement
-------------------------------------------------------

In the :ref:`previous application <lq_nsi>`, we generated income dynamics with an inverted U shape using polynomials, and placed them in an LQ framework.

It is arguably the case that this income process still contains unrealistic features.

A more common earning profile is where

#. income grows over working life, fluctuating around an increasing trend, with growth flattening off in later years

#. retirement follows, with lower but relatively stable (non-financial) income

Letting :math:`K` be the retirement date, we can express these income dynamics
by

.. math::
    :label: lq_cases

    y_t =
    \begin{cases}
    p(t) + \sigma w_{t+1} & \quad \text{if } t \leq K  \\
    s                     & \quad \text{otherwise }  \\
    \end{cases}

Here

* :math:`p(t) := m_1 t + m_2 t^2` with the coefficients :math:`m_1, m_2` chosen such that :math:`p(K) = \mu` and :math:`p(0) = p(2K)=0`

* :math:`s` is retirement income

We suppose that preferences are unchanged and given by :eq:`lq_pio`.

The budget constraint is also unchanged and given by :math:`a_{t+1} = (1 + r) a_t - c_t + y_t`.

Our aim is to solve this problem and simulate paths using the LQ techniques described in this lecture.

In fact this is a nontrivial problem, as the kink in the dynamics :eq:`lq_cases` at :math:`K` makes it very difficult to express the law of motion as a fixed-coefficient linear system.

However, we can still use our LQ methods here by suitably linking two component LQ problems.

These two LQ problems describe the consumer's behavior during her working life (``lq_working``) and retirement (``lq_retired``).

(This is possible because in the two separate periods of life, the respective income processes
[polynomial trend and constant] each fit the LQ framework)

The basic idea is that although the whole problem is not a single time-invariant LQ problem, it is
still a dynamic programming problem, and hence we can use appropriate Bellman equations at
every stage.

Based on this logic, we can

#. solve ``lq_retired`` by the usual backwards induction procedure, iterating back to the start of retirement

#. take the start-of-retirement value function generated by this process, and use it as  the terminal condition :math:`R_f` to feed into the ``lq_working`` specification

#. solve ``lq_working`` by backwards induction from this choice of :math:`R_f`, iterating back to the start of working life

This process gives the entire life-time sequence of value functions and optimal policies.

The next figure shows one simulation based on this procedure.

.. _solution_lqc_ex2_fig:

.. figure:: /_static/figures/solution_lqc_ex2.png

The full set of parameters used in the simulation is discussed in :ref:`Exercise 2 <lqc_ex2>`, where you are asked to replicate the figure.

Once again, the dominant feature observable in the simulation is consumption
smoothing.

The asset path fits well with standard life cycle theory, with dissaving early
in life followed by later saving.

Assets peak at retirement and subsequently decline.

.. _lqc_mwac:

Application 3: Monopoly with Adjustment Costs
---------------------------------------------

Consider a monopolist facing stochastic inverse demand function

.. math::

    p_t = a_0 - a_1 q_t + d_t

Here :math:`q_t` is output, and the demand shock :math:`d_t` follows

.. math::

    d_{t+1} = \rho d_t + \sigma w_{t+1}

where :math:`\{w_t\}` is iid and standard normal.

The monopolist maximizes the expected discounted sum of present and future profits

.. math::
    :label: lq_object_mp

    \mathbb E \,
    \left\{
        \sum_{t=0}^{\infty} \beta^t
        \pi_t
    \right\}
    \quad \text{where} \quad
    \pi_t := p_t q_t - c q_t - \gamma (q_{t+1} - q_t)^2

Here

* :math:`\gamma (q_{t+1} - q_t)^2` represents adjustment costs

* :math:`c` is average cost of production

This can be formulated as an LQ problem and then solved and simulated,
but first let's study the problem and try to get some intuition.

One way to start thinking about the problem is to consider what would happen
if :math:`\gamma = 0`.

Without adjustment costs there is no intertemporal trade-off, so the
monopolist will choose output to maximize current profit in each period.

It's not difficult to show that profit-maximizing output is

.. math::

    \bar q_t := \frac{a_0 - c + d_t}{2 a_1}

In light of this discussion, what we might expect for general :math:`\gamma` is that

* if :math:`\gamma` is close to zero, then :math:`q_t` will track the time path of :math:`\bar q_t` relatively closely

* if :math:`\gamma` is larger, then :math:`q_t` will be smoother than :math:`\bar q_t`, as the monopolist seeks to avoid adjustment costs

This intuition turns out to be correct.

The following figures show simulations produced by solving the corresponding LQ problem.

The only difference in parameters across the figures is the size of :math:`\gamma`.

.. figure:: /_static/figures/solution_lqc_ex3_g1.png

.. figure:: /_static/figures/solution_lqc_ex3_g10.png

.. figure:: /_static/figures/solution_lqc_ex3_g50.png

To produce these figures we converted the monopolist problem into an LQ problem.

The key to this conversion is to choose the right state --- which can be a bit of an art.

Here we take :math:`x_t = (\bar q_t \;\, q_t \;\, 1)'`, while the control is chosen as :math:`u_t = q_{t+1} - q_t`.

We also manipulated the profit function slightly.

In :eq:`lq_object_mp`, current profits are :math:`\pi_t := p_t q_t - c q_t - \gamma (q_{t+1} - q_t)^2`.

Let's now replace :math:`\pi_t` in :eq:`lq_object_mp` with :math:`\hat \pi_t := \pi_t - a_1 \bar q_t^2`.

This makes no difference to the solution, since :math:`a_1 \bar q_t^2` does not depend on the controls.

(In fact we are just adding a constant term to :eq:`\lq_object_mp`, and optimizers are not affected by constant terms)

The reason for making this substitution is that, as you will be able to
verify, :math:`\hat \pi_t` reduces to the simple quadratic

.. math::

    \hat \pi_t = -a_1 (q_t - \bar q_t)^2 - \gamma u_t^2

After negation to convert to a minimization problem, the objective becomes

.. math::
    :label: lq_object_mp2

    \min
    \mathbb E \,
        \sum_{t=0}^{\infty} \beta^t
    \left\{
        a_1 ( q_t - \bar q_t)^2 + \gamma u_t^2
    \right\}

It's now relatively straightforward to find :math:`R` and :math:`Q` such that
:eq:`lq_object_mp2` can be written as :eq:`lq_object_ih`.

Furthermore, the matrices :math:`A, B` and :math:`C` from :eq:`lq_lom`
can be found by writing down the dynamics of each element of the state.

:ref:`Exercise 3 <lqc_ex3>` asks you to complete this process, and reproduce the preceding figures.

Exercises
=========

.. _lqc_ex1:

Exercise 1
----------

Replicate the figure with polynomial income :ref:`shown above <solution_lqc_ex1_fig>`.

The parameters are :math:`r = 0.05, \beta = 1 / (1 + r), \bar c = 1.5,  \mu = 2, \sigma = 0.15, T = 50` and :math:`q = 10^4`.

.. _lqc_ex2:

Exercise 2
----------

Replicate the figure on work and retirement :ref:`shown above <solution_lqc_ex2_fig>`.

The parameters are :math:`r = 0.05, \beta = 1 / (1 + r), \bar c = 4,  \mu = 4, \sigma = 0.35, K = 40, T = 60, s = 1` and :math:`q = 10^4`.

To understand the overall procedure, carefully read the section containing that figure.

Some hints are as follows:

First, in order to make our approach work, we must ensure that both LQ problems have the same state variables and control.

As with previous applications, the control can be set to :math:`u_t = c_t - \bar c`.

For ``lq_working``, :math:`x_t, A, B, C` can be chosen as in :eq:`lq_lowmc3`.

* Recall that :math:`m_1, m_2` are chosen so that :math:`p(K) = \mu` and :math:`p(2K)=0`.

For ``lq_retired``, use the same definition of :math:`x_t` and :math:`u_t`, but modify :math:`A, B, C` to correspond to constant income :math:`y_t = s`.

For ``lq_retired``, set preferences as in :eq:`lq_4sp`.

For ``lq_working``, preferences are the same, except that :math:`R_f` should
be replaced by the final value function that emerges from iterating ``lq_retired``
back to the start of retirement.

With some careful footwork, the simulation can be generated by patching
together the simulations from these two separate models.

.. _lqc_ex3:

Exercise 3
----------

Reproduce the figures from the monopolist application :ref:`given above <lqc_mwac>`.

For parameters, use :math:`a_0 = 5, a_1 = 0.5, \sigma = 0.15, \rho = 0.9,
\beta = 0.95` and :math:`c = 2`, while :math:`\gamma` varies between 1 and 50
(see figures).

Solutions
=========

Exercise 1
----------

Here’s one solution.

We use some fancy plot commands to get a certain style — feel free to
use simpler ones.

The model is an LQ permanent income / life-cycle model with hump-shaped
income.

.. math::     y_t = m_1 t + m_2 t^2 + \sigma w_{t+1}

where :math:`\{w_t\}` is iid :math:`N(0, 1)` and the coefficients
:math:`m_1` and :math:`m_2` are chosen so that
:math:`p(t) = m_1 t + m_2 t^2` has an inverted U shape with

-  :math:`p(0) = 0, p(T/2) = \mu`, and
-  :math:`p(T) = 0`.

.. code-block:: julia

    # model parameters
    r = 0.05
    β = 1 / (1 + r)
    T = 50
    c̄ = 1.5
    σ = 0.15
    μ = 2
    q = 1e4
    m1 = T * (μ / (T / 2)^2)
    m2 = -(μ / (T / 2)^2)

    # formulate as an LQ problem
    Q = 1.0
    R = zeros(4, 4)
    Rf = zeros(4, 4); Rf[1, 1] = q
    A = [1 + r -c̄ m1 m2;
         0     1      0  0;
         0     1      1  0;
         0     1      2  1]
    B = [-1.0, 0.0, 0.0, 0.0]
    C = [σ, 0.0, 0.0, 0.0]

    # compute solutions and simulate
    lq = QuantEcon.LQ(Q, R, A, B, C; bet = β, capT = T, rf = Rf)
    x0 = [0.0, 1, 0, 0]
    xp, up, wp = compute_sequence(lq, x0)

    # convert results back to assets, consumption and income
    ap = vec(xp[1, 1:end]) # assets
    c = vec(up .+ c̄) # consumption
    time = 1:T
    income = σ * vec(wp[1, 2:end]) + m1 * time + m2 * time.^2 # income

    # plot results
    p1 = plot(Vector[income, ap, c, zeros(T + 1)],
              lab = ["non-financial income" "assets" "consumption" ""],
              color = [:orange :blue :green :black],
              xaxis = "Time", layout = (2,1),
              bottom_margin = 20mm, size = (600, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test ap[20] ≈ -8.123970759436794
    end

Exercise 2
----------

This is a permanent income / life-cycle model with polynomial growth in
income over working life followed by a fixed retirement income. The
model is solved by combining two LQ programming problems as described in
the lecture.

.. code-block:: julia

    # model parameters
    r = 0.05
    β = 1/(1 + r)
    T = 60
    K = 40
    c̄ = 4
    σ = 0.35
    μ = 4
    q = 1e4
    s = 1
    m1 = 2 * μ / K
    m2 = - μ / K^2

    # formulate LQ problem 1 (retirement)
    Q = 1.0
    R = zeros(4, 4)
    Rf = zeros(4, 4); Rf[1, 1] = q
    A = [1+r  s-c̄ 0 0;
         0     1      0 0;
         0     1      1 0;
         0     1      2 1]
    B = [-1.0, 0, 0, 0]
    C = zeros(4)

    # initialize LQ instance for retired agent
    lq_retired = QuantEcon.LQ(Q, R, A, B, C; bet = β, capT = T - K, rf = Rf)

    # since update_values!() changes its argument in place, we need another identical instance
    # just to get the correct value function
    lq_retired_proxy = QuantEcon.LQ(Q, R, A, B, C; bet = β, capT = T - K, rf = Rf)

    # iterate back to start of retirement, record final value function
    for i in 1:(T - K)
        update_values!(lq_retired_proxy)
    end
    Rf2 = lq_retired_proxy.P

    # formulate LQ problem 2 (working life)
    Q = 1.0
    R = zeros(4, 4)
    A = [1 + r -c̄ m1 m2;
         0     1      0  0;
         0     1      1  0;
         0     1      2  1]
    B = [-1.0, 0, 0, 0]
    C = [σ, 0, 0, 0]

    # set up working life LQ instance with terminal Rf from lq_retired
    lq_working = QuantEcon.LQ(Q, R, A, B, C; bet = β, capT = K, rf = Rf2)

    # simulate working state / control paths
    x0 = [0.0, 1, 0, 0]
    xp_w, up_w, wp_w = compute_sequence(lq_working, x0)

    # simulate retirement paths (note the initial condition)
    xp_r, up_r, wp_r = compute_sequence(lq_retired, xp_w[:, end])

    # convert results back to assets, consumption and income
    xp = [xp_w xp_r[:, 2:end]]
    assets = vec(xp[1, :]) # assets

    up = [up_w up_r]
    c = vec(up .+ c̄) # consumption

    time = 1:K
    income_w = σ * vec(wp_w[1, 2:K+1]) + m1 .* time + m2 .* time.^2 # income
    income_r = ones(T - K) * s
    income = [income_w; income_r]

    # plot results
    p2 = plot([income, assets, c, zeros(T + 1)],
              lab = ["non-financial income" "assets" "consumption" ""],
              color = [:orange :blue :green :black],
              xaxis = "Time", layout = (2, 1),
              bottom_margin = 20mm, size = (600, 600))

.. code-block:: julia
    :class: test

    @testset begin
        @test income[20] ≈ 3.0809720173965855
    end

Exercise 3
----------

The first task is to find the matrices :math:`A, B, C, Q, R` that define
the LQ problem.

Recall that :math:`x_t = (\bar q_t \;\, q_t \;\, 1)'`, while
:math:`u_t = q_{t+1} - q_t`.

Letting :math:`m_0 := (a_0 - c) / 2a_1` and :math:`m_1 := 1 / 2 a_1`, we
can write :math:`\bar q_t = m_0 + m_1 d_t`, and then, with some
manipulation

.. math::

       \bar q_{t+1} = m_0 (1 - \rho) + \rho \bar q_t + m_1 \sigma w_{t+1}

By our definition of :math:`u_t`, the dynamics of :math:`q_t` are
:math:`q_{t+1} = q_t + u_t`.

Using these facts you should be able to build the correct
:math:`A, B, C` matrices (and then check them against those found in the
solution code below).

Suitable :math:`R, Q` matrices can be found by inspecting the objective
function, which we repeat here for convenience:

.. math::

       \min
       \mathbb E \,
       \left\{
           \sum_{t=0}^{\infty} \beta^t
           a_1 ( q_t - \bar q_t)^2 + \gamma u_t^2
       \right\}

Our solution code is

.. code-block:: julia

    # model parameters
    a0 = 5.0
    a1 = 0.5
    σ = 0.15
    ρ = 0.9
    γ = 1.0
    β = 0.95
    c = 2.0
    T = 120

    # useful constants
    m0 = (a0 - c) / (2 * a1)
    m1 = 1 / (2 * a1)

    # formulate LQ problem
    Q = γ
    R = [a1 -a1 0; -a1 a1 0; 0 0 0]
    A = [ρ 0 m0 * (1 - ρ); 0 1 0; 0 0 1]

    B = [0.0, 1, 0]
    C = [m1 * σ, 0, 0]

    lq = QuantEcon.LQ(Q, R, A, B, C; bet = β)

    # simulate state / control paths
    x0 = [m0, 2, 1]
    xp, up, wp = compute_sequence(lq, x0, 150)
    q̄ = vec(xp[1, :])
    q = vec(xp[2, :])

    # plot simulation results
    p3 = plot(eachindex(q), [q̄ q],
              lab = ["q bar" "q"],
              color = [:black :blue],
              xaxis = "Time", title = "Dynamics with γ = $γ",
              bottom_margin = 20mm, top_margin = 10mm,
              size = (700, 500))

.. code-block:: julia
    :class: test

    @testset begin
        @test xp[20] ≈ 2.8378651501210808
        @test q̄[25] ≈ 2.518687537862516
    end
    