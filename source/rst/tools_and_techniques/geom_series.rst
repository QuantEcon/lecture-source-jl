.. _geom_series:

.. include:: /_static/includes/header.raw

.. highlight:: julia

*****************************************
Geometric Series for Elementary Economics
*****************************************

.. contents:: :depth: 2

Overview
========


The lecture describes important ideas in economics that use the mathematics of geometric series.

Among these are

-  the Keynesian **multiplier**

-  the money **multiplier** that prevails in fractional reserve banking
   systems

-  interest rates and present values of streams of payouts from assets

(As we shall see below, the term **multiplier** comes down to meaning **sum of a convergent geometric series**)



These and other applications prove the truth of the wise crack that

.. epigraph::

    "in economics, a little knowledge of geometric series goes a long way "

Setup
------------------

Below we'll use the following imports:

.. literalinclude:: /_static/includes/deps_generic.jl
      :class: hide-output

.. code-block:: julia

  using Plots

.. code-block:: julia 
    :class: test 

    using Test 

Key Formulas
============

To start, let :math:`c` be a real number that lies strictly between
:math:`-1` and :math:`1`.

-  We often write this as :math:`c \in (-1,1)`.

-  Here :math:`(-1,1)` denotes the collection of all real numbers that
   are strictly less than :math:`1` and strictly greater than :math:`-1`.

-  The symbol :math:`\in` means *in* or *belongs to the set after the symbol*.

We want to evaluate geometric series of two types -- infinite and finite.

Infinite Geometric Series
-------------------------

The first type of geometric that interests us is the infinite series

.. math:: 1 + c + c^2 + c^3 + \cdots

Where :math:`\cdots` means that the series continues without limit.

The key formula is

.. math::
   :label: infinite

   1 + c + c^2 + c^3 + \cdots = \frac{1}{1 -c }

To prove key formula :eq:`infinite`, multiply both sides  by :math:`(1-c)` and verify
that if :math:`c \in (-1,1)`, then the outcome is the
equation :math:`1 = 1`.

Finite Geometric Series
-----------------------

The second series that interests us is the finite geometric series

.. math:: 1 + c + c^2 + c^3 + \cdots + c^T

where :math:`T` is a positive integer.

The key formula here is

.. math:: 1 + c + c^2 + c^3 + \cdots + c^T  = \frac{1 - c^{T+1}}{1-c}

**Remark:** The above formula works for any value of the scalar
:math:`c`. We don't have to restrict :math:`c` to be in the
set :math:`(-1,1)`.




We now move on to describe some famous economic applications of
geometric series.



Example: The Money Multiplier in Fractional Reserve Banking
===========================================================

In a fractional reserve banking system, banks hold only a fraction
:math:`r \in (0,1)` of cash behind each **deposit receipt** that they
issue

*  In recent times

   -  cash consists of pieces of paper issued by the government and
      called dollars or pounds or :math:`\ldots`

   -  a *deposit* is a balance in a checking or savings account that
      entitles the owner to ask the bank for immediate payment in cash

*  When the UK and France and the US were on either a gold or silver
   standard (before 1914, for example)

   -  cash was a gold or silver coin

   -  a *deposit receipt* was a *bank note* that the bank promised to
      convert into gold or silver on demand; (sometimes it was also a
      checking or savings account balance)

Economists and financiers often define the **supply of money** as an
economy-wide sum of **cash** plus **deposits**.

In a **fractional reserve banking system** (one in which the reserve
ratio :math:`r` satisfying :math:`0 < r < 1`), **banks create money** by issuing deposits *backed* by fractional reserves plus loans that they make to their customers.

A geometric series is a key tool for understanding how banks create
money (i.e., deposits) in a fractional reserve system.

The geometric series formula :eq:`infinite` is at the heart of the classic model of the money creation process -- one that leads us to the celebrated
**money multiplier**.



A Simple Model
--------------

There is a set of banks named :math:`i = 0, 1, 2, \ldots`.

Bank :math:`i`'s loans :math:`L_i`, deposits :math:`D_i`, and
reserves :math:`R_i` must satisfy the balance sheet equation (because
**balance sheets balance**):

.. math:: L_i + R_i = D_i

The left side of the above equation is the sum of the bank's **assets**,
namely, the loans :math:`L_i` it has outstanding plus its reserves of
cash :math:`R_i`.

The right side records bank :math:`i`'s liabilities,
namely, the deposits :math:`D_i` held by its depositors; these are
IOU's from the bank to its depositors in the form of either checking
accounts or savings accounts (or before 1914, bank notes issued by a
bank stating promises to redeem note for gold or silver on demand).

.. TO REMOVE:
.. Dongchen: is there a way to add a little balance sheet here?
.. with assets on the left side and liabilities on the right side?

Each bank :math:`i` sets its reserves to satisfy the equation

.. math::
  :label: reserves

  R_i = r D_i

where :math:`r \in (0,1)` is its **reserve-deposit ratio** or **reserve
ratio** for short

-  the reserve ratio is either set by a government or chosen by banks
   for precautionary reasons

Next we add a theory stating that bank :math:`i+1`'s deposits depend
entirely on loans made by bank :math:`i`, namely

.. math::
  :label: deposits

  D_{i+1} = L_i

Thus, we can think of the banks as being arranged along a line with
loans from bank :math:`i` being immediately deposited in :math:`i+1`

-  in this way, the debtors to bank :math:`i` become creditors of
   bank :math:`i+1`

Finally, we add an *initial condition* about an exogenous level of bank
:math:`0`'s deposits

.. math:: D_0 \ \text{ is given exogenously}

We can think of :math:`D_0` as being the amount of cash that a first
depositor put into the first bank in the system, bank number :math:`i=0`.

Now we do a little algebra.

Combining equations :eq:`reserves` and :eq:`deposits` tells us that

.. math::
  :label: fraction

  L_i = (1-r) D_i

This states that bank :math:`i` loans a fraction :math:`(1-r)` of its
deposits and keeps a fraction :math:`r` as cash reserves.

Combining equation :eq:`fraction` with equation :eq:`deposits` tells us that

.. math:: D_{i+1} = (1-r) D_i  \ \text{ for } i \geq 0

which implies that

.. math::
  :label: geomseries

  D_i = (1 - r)^i D_0  \ \text{ for } i \geq 0

Equation :eq:`geomseries` expresses :math:`D_i` as the :math:`i` th term in the
product of :math:`D_0` and the geometric series

.. math::  1, (1-r), (1-r)^2, \cdots

Therefore, the sum of all deposits in our banking system
:math:`i=0, 1, 2, \ldots` is

.. math::
  :label: sumdeposits

  \sum_{i=0}^\infty (1-r)^i D_0 =  \frac{D_0}{1 - (1-r)} = \frac{D_0}{r}


Money Multiplier
----------------

The **money multiplier** is a number that tells the multiplicative
factor by which an exogenous injection of cash into bank :math:`0` leads
to an increase in the total deposits in the banking system.

Equation :eq:`sumdeposits` asserts that the **money multiplier** is
:math:`\frac{1}{r}`

-  An initial deposit of cash of :math:`D_0` in bank :math:`0` leads
   the banking system to create total deposits of :math:`\frac{D_0}{r}`.

-  The initial deposit :math:`D_0` is held as reserves, distributed
   throughout the banking system according to :math:`D_0 = \sum_{i=0}^\infty R_i`.

.. Dongchen: can you think of some simple Python examples that
.. illustrate how to create sequences and so on? Also, some simple
.. experiments like lowering reserve requirements? Or others you may suggest?


Example: The Keynesian Multiplier
=================================

The famous economist John Maynard Keynes and his followers created a
simple model intended to determine national income :math:`y` in
circumstances in which

-  there are substantial unemployed resources, in particular **excess
   supply** of labor and capital

-  prices and interest rates fail to adjust to make aggregate **supply
   equal demand** (e.g., prices and interest rates are frozen)

-  national income is entirely determined by aggregate demand


Static Version
--------------


An elementary Keynesian model of national income determination consists
of three equations that describe aggregate demand for :math:`y` and its
components.

The first equation is a national income identity asserting that
consumption :math:`c` plus investment :math:`i` equals national income
:math:`y`:

.. math:: c+ i = y

The second equation is a Keynesian consumption function asserting that
people consume a fraction :math:`b \in (0,1)` of their income:

.. math:: c = b y

The fraction :math:`b \in (0,1)` is called the **marginal propensity to
consume**.

The fraction :math:`1-b \in (0,1)` is called the **marginal propensity
to save**.

The third equation simply states that investment is exogenous at level
:math:`i`.

- *exogenous* means *determined outside this model*.

Substituting the second equation into the first gives :math:`(1-b) y = i`.

Solving this equation for :math:`y` gives

.. math:: y = \frac{1}{1-b} i

The quantity :math:`\frac{1}{1-b}` is called the **investment
multiplier** or simply the **multiplier**.

Applying the formula for the sum of an infinite geometric series, we can
write the above equation as

.. math:: y = i \sum_{t=0}^\infty b^t

where :math:`t` is a nonnegative integer.

So we arrive at the following equivalent expressions for the multiplier:

.. math:: \frac{1}{1-b} =   \sum_{t=0}^\infty b^t

The expression :math:`\sum_{t=0}^\infty b^t` motivates an interpretation
of the multiplier as the outcome of a dynamic process that we describe
next.

Dynamic Version
---------------

We arrive at a dynamic version by interpreting the nonnegative integer
:math:`t` as indexing time and changing our specification of the
consumption function to take time into account

-  we add a one-period lag in how income affects consumption

We let :math:`c_t` be consumption at time :math:`t` and :math:`i_t` be
investment at time :math:`t`.

We modify our consumption function to assume the form

.. math:: c_t = b y_{t-1}

so that :math:`b` is the marginal propensity to consume (now) out of
last period's income.

We begin with an initial condition stating that

.. math:: y_{-1} = 0

We also assume that

.. math:: i_t = i \ \ \textrm {for all }  t \geq 0

so that investment is constant over time.

It follows that

.. math:: y_0 = i + c_0 = i + b y_{-1} =  i

and

.. math:: y_1 = c_1 + i = b y_0 + i = (1 + b) i

and

.. math:: y_2 = c_2 + i = b y_1 + i = (1 + b + b^2) i

and more generally

.. math:: y_t = b y_{t-1} + i = (1+ b + b^2 + \cdots + b^t) i

or

.. math:: y_t = \frac{1-b^{t+1}}{1 -b } i

Evidently, as :math:`t \rightarrow + \infty`,

.. math:: y_t \rightarrow \frac{1}{1-b} i

**Remark 1:** The above formula is often applied to assert that an
exogenous increase in investment of :math:`\Delta i` at time :math:`0`
ignites a dynamic process of increases in national income by amounts

.. math:: \Delta i, (1 + b )\Delta i, (1+b + b^2) \Delta i , \cdots

at times :math:`0, 1, 2, \ldots`.

**Remark 2** Let :math:`g_t` be an exogenous sequence of government
expenditures.

If we generalize the model so that the national income identity
becomes

.. math:: c_t + i_t + g_t  = y_t

then a version of the preceding argument shows that the **government
expenditures multiplier** is also :math:`\frac{1}{1-b}`, so that a
permanent increase in government expenditures ultimately leads to an
increase in national income equal to the multiplier times the increase
in government expenditures.

.. Dongchen: can you think of some simple Python things to add to
.. illustrate basic concepts, maybe the idea of a "difference equation" and how we solve it?


Example: Interest Rates and Present Values
==========================================

We can apply our formula for geometric series to study how interest
rates affect values of streams of dollar payments that extend over time.

We work in discrete time and assume that :math:`t = 0, 1, 2, \ldots`
indexes time.

We let :math:`r \in (0,1)` be a one-period **net nominal interest rate**

-  if the nominal interest rate is :math:`5` percent,
   then :math:`r= .05`

A one-period **gross nominal interest rate** :math:`R` is defined as

.. math:: R = 1 + r \in (1, 2)

-  if :math:`r=.05`, then :math:`R = 1.05`

**Remark:** The gross nominal interest rate :math:`R` is an **exchange
rate** or **relative price** of dollars at between times :math:`t` and
:math:`t+1`. The units of :math:`R` are dollars at time :math:`t+1` per
dollar at time :math:`t`.

When people borrow and lend, they trade dollars now for dollars later or
dollars later for dollars now.

The price at which these exchanges occur is the gross nominal interest
rate.

-  If I sell :math:`x` dollars to you today, you pay me :math:`R x`
   dollars tomorrow.

-  This means that you borrowed :math:`x` dollars for me at a gross
   interest rate :math:`R` and a net interest rate :math:`r`.

We assume that the net nominal interest rate :math:`r` is fixed over
time, so that :math:`R` is the gross nominal interest rate at times
:math:`t=0, 1, 2, \ldots`.

Two important geometric sequences are

.. math::
  :label: geom1

  1, R, R^2, \cdots

and

.. math::
  :label: geom2

  1, R^{-1}, R^{-2}, \cdots

Sequence :eq:`geom1` tells us how dollar values of an investment **accumulate**
through time.

Sequence :eq:`geom2` tells us how to **discount** future dollars to get their
values in terms of today's dollars.

Accumulation
------------

Geometric sequence :eq:`geom1` tells us how one dollar invested and re-invested
in a project with gross one period nominal rate of return accumulates

-  here we assume that net interest payments are reinvested in the
   project

-  thus, :math:`1` dollar invested at time :math:`0` pays interest
   :math:`r` dollars after one period, so we have :math:`r+1 = R`
   dollars at time\ :math:`1`

-  at time :math:`1` we reinvest :math:`1+r =R` dollars and receive interest
   of :math:`r R` dollars at time :math:`2` plus the *principal*
   :math:`R` dollars, so we receive :math:`r R + R = (1+r)R = R^2`
   dollars at the end of period :math:`2`

-  and so on

Evidently, if we invest :math:`x` dollars at time :math:`0` and
reinvest the proceeds, then the sequence

.. math:: x , xR , x R^2, \cdots

tells how our account accumulates at dates :math:`t=0, 1, 2, \ldots`.

Discounting
-----------

Geometric sequence :eq:`geom2` tells us how much future dollars are worth in terms of today's dollars.

Remember that the units of :math:`R` are dollars at :math:`t+1` per
dollar at :math:`t`.

It follows that

-  the units of :math:`R^{-1}` are dollars at :math:`t` per dollar at :math:`t+1`

-  the units of :math:`R^{-2}` are dollars at :math:`t` per dollar at :math:`t+2`

-  and so on; the units of :math:`R^{-j}` are dollars at :math:`t` per
   dollar at :math:`t+j`

So if someone has a claim on :math:`x` dollars at time :math:`t+j`, it
is worth :math:`x R^{-j}` dollars at time :math:`t` (e.g., today).



Application to Asset Pricing
----------------------------

A **lease** requires a payments stream of :math:`x_t` dollars at
times :math:`t = 0, 1, 2, \ldots` where

.. math::  x_t = G^t x_0

where :math:`G = (1+g)` and :math:`g \in (0,1)`.

Thus, lease payments increase at :math:`g` percent per period.

For a reason soon to be revealed, we assume that :math:`G < R`.

The **present value** of the lease is

.. math::

    \begin{aligned} p_0  & = x_0 + x_1/R + x_2/(R^2) + \ddots \\
                     & = x_0 (1 + G R^{-1} + G^2 R^{-2} + \cdots ) \\
                     & = x_0 \frac{1}{1 - G R^{-1}} \end{aligned}

where the last line uses the formula for an infinite geometric series.

Recall that :math:`R = 1+r` and :math:`G = 1+g` and that :math:`R > G`
and :math:`r > g` and that :math:`r` and\ :math:`g` are typically small
numbers, e.g., .05 or .03.

Use the Taylor series of :math:`\frac{1}{1+r}` about :math:`r=0`,
namely,

.. math:: \frac{1}{1+r} = 1 - r + r^2 - r^3 + \cdots

and the fact that :math:`r` is small to approximate
:math:`\frac{1}{1+r} \approx 1 - r`.

Use this approximation to write :math:`p_0` as

.. math::

    \begin{aligned}
    p_0 &= x_0 \frac{1}{1 - G R^{-1}} \\
    &= x_0 \frac{1}{1 - (1+g) (1-r) } \\
    &= x_0 \frac{1}{1 - (1+g - r - rg)} \\
    & \approx x_0 \frac{1}{r -g }
   \end{aligned}

where the last step uses the approximation :math:`r g \approx 0`.

The approximation

.. math:: p_0 = \frac{x_0 }{r -g }

is known as the **Gordon formula** for the present value or current
price of an infinite payment stream :math:`x_0 G^t` when the nominal
one-period interest rate is :math:`r` and when :math:`r > g`.


We can also extend the asset pricing formula so that it applies to finite leases.

Let the payment stream on the lease now be :math:`x_t` for :math:`t= 1,2, \dots,T`, where again

.. math:: x_t = G^t x_0

The present value of this lease is:

.. math:: \begin{aligned} \begin{split}p_0&=x_0 + x_1/R  + \dots +x_T/R^T \\ &= x_0(1+GR^{-1}+\dots +G^{T}R^{-T}) \\ &= \frac{x_0(1-G^{T+1}R^{-(T+1)})}{1-GR^{-1}}  \end{split}\end{aligned}

Applying the Taylor series to :math:`R^{-(T+1)}` about :math:`r=0` we get:

.. math:: \frac{1}{(1+r)^{T+1}}= 1-r(T+1)+\frac{1}{2}r^2(T+1)(T+2)+\dots \approx 1-r(T+1)

Similarly, applying the Taylor series to :math:`G^{T+1}` about :math:`g=0`:

.. math:: (1+g)^{T+1} = 1+(T+1)g(1+g)^T+(T+1)Tg^2(1+g)^{T-1}+\dots \approx 1+ (T+1)g

Thus, we get the following approximation:

.. math:: p_0 =\frac{x_0(1-(1+(T+1)g)(1-r(T+1)))}{1-(1-r)(1+g) }

Expanding:

.. math::  \begin{aligned} p_0 &=\frac{x_0(1-1+(T+1)^2 rg -r(T+1)+g(T+1))}{1-1+r-g+rg}  \\&=\frac{x_0(T+1)((T+1)rg+r-g)}{r-g+rg} \\ &\approx \frac{x_0(T+1)(r-g)}{r-g}+\frac{x_0rg(T+1)}{r-g}\\ &= x_0(T+1) + \frac{x_0rg(T+1)}{r-g}  \end{aligned}

We could have also approximated by removing the second term
:math:`rgx_0(T+1)` when :math:`T` is relatively small compared to
:math:`1/(rg)` to get :math:`x_0(T+1)` as in the finite stream
approximation.

We will plot the true finite stream present-value and the two
approximations, under different values of :math:`T`, and :math:`g` and :math:`r` in Julia.

First we plot the true finite stream present-value after computing it
below

.. code-block:: julia

    # True present value of a finite lease
    function finite_lease_pv(T, g, r, x_0)
        G = 1 + g
        R = 1 + r
        return (x_0 * (1 - G^(T + 1)* R^(-(T + 1)))) / (1 - G * R^(-1))
    end

    # First approximation for our finite lease
    finite_lease_pv_approx_f(T, g, r, x_0) = x_0 * (T + 1) + x_0 * r * g * (T + 1) / (r - g)

    # Second approximation for our finite lease
    finite_lease_pv_approx_s(T, g, r, x_0) = x_0 * (T + 1)

    # Infinite lease
    function infinite_lease(g, r, x_0)
        G = 1 + g
        R = 1 + r
        return x_0 / (1 - G * R^(-1))
    end


Now that we have test run our functions, we can plot some outcomes.

First we study the quality of our approximations

.. code-block:: julia

    g = 0.02
    r = 0.03
    x_0 = 1
    T_max = 50
    T = 0:T_max
    y_1 = finite_lease_pv.(T, g, r, x_0)
    y_2 = finite_lease_pv_approx_f.(T, g, r, x_0)
    y_3 = finite_lease_pv_approx_s.(T, g, r, x_0)

    plot(T, y_1, label="True T-period Lease PV", title= "Finite Lease Present Value T Periods Ahead", xlabel = "T Periods Ahead", ylabel = "Present Value, p₀")
    plot!(T, y_2, label="T-period Lease First-order Approx.")
    plot!(T, y_3, label="T-period Lease First-order Approx. adj.")
    plot!(legend = :topleft)

.. code-block:: julia
    :class: test

    @testset begin
    @test y_1[4] ≈ 3.942123696037542
    @test y_2[4] ≈ 4.24
    @test y_3[4] ≈ 4
    end

Evidently our approximations perform well for small values of :math:`T`.

However, holding :math:`g` and r fixed, our approximations deteriorate as :math:`T` increases.

Next we compare the infinite and finite duration lease present values
over different lease lengths :math:`T`.

.. code-block:: julia

    # Convergence of infinite and finite
    T_max = 1000
    T = 0:T_max
    y_1 = finite_lease_pv.(T, g, r, x_0)
    y_2 = ones(T_max+1) .* infinite_lease(g, r, x_0)
    plot(T, y_1, label="T-period lease PV", title= "Infinite and Finite Lease Present Value T Periods Ahead", xlabel = "T Periods Ahead", ylabel = "Present Value, p0")
    plot!(T, y_2, linestyle = :dash, label="Infinite lease PV")
    plot!(legend = :bottomright)

.. code-block:: julia
    :class: test

    @testset begin
       @test y_1[4] ≈ 3.942123696037542
       @test y_2[4] ≈ 103.00000000000004
    end


The above graphs shows how as duration :math:`T \rightarrow +\infty`,
the value of a lease of duration :math:`T` approaches the value of a
perpetual lease.

Now we consider two different views of what happens as :math:`r` and
:math:`g` covary

.. code-block:: julia

    T_max = 10
    T=0:T_max

    # r >> g, much bigger than g
    r = 0.9
    g = 0.4
    plot(finite_lease_pv.(T, g, r, x_0), label="r >> g", title= "Value of lease of length T", xlabel = "T periods ahead", ylabel = "Present Value, p0")

    # r > g
    r = 0.5
    g = 0.4
    plot!(finite_lease_pv.(T, g, r, x_0), label="r > g", color="green")

    # r ~ g, not defined when r = g, but approximately goes to straight
    # line with slope 1
    r = 0.4001
    g = 0.4
    plot!(finite_lease_pv.(T, g, r, x_0), label="r ~ g", color="orange")

    # r < g
    r = 0.4
    g = 0.5
    plot!(finite_lease_pv.(T, g, r, x_0), label="r < g", color="red")
    plot!(legend = :topleft)

The above graphs gives a big hint for why the condition :math:`r > g` is
necessary if a lease of length :math:`T = +\infty` is to have finite
value.

For fans of 3-d graphs the same point comes through in the following
graph.

If you aren't enamored of 3-d graphs, feel free to skip the next
visualization!

.. code-block:: julia

    T = 3
    r = 0.01:0.005:0.985
    g = 0.011:0.005:0.986
    N = length(r)

    rr = reshape(repeat(r, N), N, N)'
    gg = repeat(g, 1, N)
    z = finite_lease_pv.(T, gg, rr, x_0)
    plot(r, g, z, st=:surface, c=:coolwarm,title= "Three Period Lease PV with varying g and r", xlabel = "r", ylabel = "g")

.. code-block:: julia
    :class: test

    @testset begin
        @test z[4] ≈ 4.096057303642319
    end

.. We can use a little calculus to study how the present value :math:`p_0`
.. of a lease varies with :math:`r` and :math:`g`.
..
.. We will use a library called `SymPy <https://www.sympy.org/>`__.
..
.. SymPy enables us to do symbolic math calculations including
.. computing derivatives of algebraic equations.
..
.. We will illustrate how it works by creating a symbolic expression that
.. represents our present value formula for an infinite lease.
..
.. After that, we'll use SymPy to compute derivatives
..
.. .. code-block:: julia
..
..     # Creates algebraic symbols that can be used in an algebraic expression
..     g, r, x0 = symbols("g, r, x0")
..     G = (1 + g)
..     R = (1 + r)
..     p0 = x0 / (1 - G * R ^ (-1))
..     print("Our formula is:")
..     p0
..
.. .. code-block:: julia
..
..     print("dp0 / dg is:")
..     dp_dg = diff(p0, g)
..     dp_dg
..
.. .. code-block:: julia
..
..     print("dp0 / dr is:")
..     dp_dr = diff(p0, r)
..     dp_dr

..     We can see that for :math:`\frac{\partial p_0}{\partial r}<0` as long as
..     :math:`r>g`, :math:`r>0` and :math:`g>0` and :math:`x_0` is positive,
..     this equation will always be negative.

..     Similarly, :math:`\frac{\partial p_0}{\partial g}>0` as long as :math:`r>g`, :math:`r>0` and :math:`g>0` and :math:`x_0` is positive, this equation
..     will always be postive.

Back to the Keynesian Multiplier
================================

We will now go back to the case of the Keynesian multiplier and plot the
time path of :math:`y_t`, given that consumption is a constant fraction
of national income, and investment is fixed.

.. code-block:: julia

    # Function that calculates a path of y
    function calculate_y(i, b, g, T, y_init)
        y = zeros(T+1)
        y[1] = i + b * y_init + g
        for t = 2:(T+1)
            y[t] = b * y[t-1] + i + g
        end
        return y
    end

    # Initial values
    i_0 = 0.3
    g_0 = 0.3
    # 2/3 of income goes towards consumption
    b = 2/3
    y_init = 0
    T = 100

    plot(0:T, calculate_y(i_0, b, g_0, T, y_init), title= "Path of Aggregate Output Over Time", xlabel = "t", label = "yt")

    # Output predicted by geometric series
    hline!([i_0 / (1 - b) + g_0 / (1 - b)], linestyle=:dash, seriestype="hline", label = "Predicted")
    plot!(legend = :bottomright)

.. code-block:: julia
    :class: test

    @testset begin
        @test calculate_y(i_0, b, g_0, T, y_init)[4] ≈ 1.4444444444444444
    end

In this model, income grows over time, until it gradually converges to
the infinite geometric series sum of income.

We now examine what will
happen if we vary the so-called **marginal propensity to consume**,
i.e., the fraction of income that is consumed

.. code-block:: julia

    # Changing fraction of consumption
    b_0 = 1/3
    b_1 = 2/3
    b_2 = 5/6
    b_3 = 0.9

    plt = plot(title= "Changing Consumption as a Fraction of Income", xlabel = "t", ylabel = "yt")
    x = 0:T
    for b in [b_0, b_1, b_2, b_3]
        y = calculate_y(i_0, b, g_0, T, y_init)
        plot!(plt, x, y, label=string("b= ",round(b,digits=2)))
    end
    plot!(plt, legend = :bottomright)


Increasing the marginal propensity to consumer :math:`b` increases the
path of output over time

.. below we need to set b=0.9 which does not happen in python
.. since python changes value by reference in the for loop

.. code-block:: julia

    x = 0:T
    b = 0.9
    y_0 = calculate_y(i_0, b, g_0, T, y_init)
    l = @layout [a ; b]

    # Changing initial investment:
    i_1 = 0.4
    y_1 = calculate_y(i_1, b, g_0, T, y_init)
    plt_1 = plot(x,y_0, label = "i=0.3", linestyle= :dash, title= "An Increase in Investment on Output", xlabel = "t", ylabel = "y_t")
    plot!(plt_1, x, y_1, label = "i=0.4")
    plot!(plt_1, legend = :bottomright)

    # Changing government spending
    g_1 = 0.4
    y_1 = calculate_y(i_0, b, g_1, T, y_init)
    plt_2 = plot(x, y_0, label = "g=0.3", linestyle= :dash, title= "An Increase in Investment on Output", xlabel = "t", ylabel = "y_t")
    plot!(plt_2, x, y_1, label="g=0.4", linestyle=:dash)
    plot!(plt_2, legend = :bottomright)

    plot(plt_1, plt_2, layout = l)

.. code-block:: julia
    :class: test

    @testset begin
        @test y_1[2] ≈ 1.33
        @test y_1[10] ≈ 4.5592509193
    end

Notice here, whether government spending increases from 0.3 to 0.4 or
investment increases from 0.3 to 0.4, the shifts in the graphs are
identical.
