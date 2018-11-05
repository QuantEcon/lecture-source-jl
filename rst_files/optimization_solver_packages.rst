.. _optimization_solver_packages:

.. include:: /_static/includes/lecture_howto_jl.raw

************************************************
Solvers, Optimizers, and Auto-differentiation
************************************************

.. contents:: :depth: 2

Overview
============

In this lecture we introduce a few of the Julia libraries that we've found particularly useful for quantitative work in economics

Setup
------------------

.. literalinclude:: /_static/includes/alldeps.jl

Using AD
=====================

Automatic differentiation (sometimes called algorithmic differentiation) is a crucial way to increase the performance of both estimation and solution methods

There are essentially three four ways to calculate the gradient or jacobian on a computer

* Calculation by hand

    * Where possible, you can calculate the derivative on "pen-and-paper" and potentially simplify the expression
    * Sometimes, though not always, the most accurate and fastest option if there are algebraic simplifications
    * The algebra is error prone for non-trivial setups

* Finite differences

    * Evaluate the function at least ``N`` times to get the gradient, jacobians are even worse
    * Large :math:`\Delta` is numerically stable but inaccurate, too small of :math:`\Delta` is numerically unstable but more accurate
    * Avoid if you can, and use prepackaged calls otherwise (to get a good choice of :math:`\Delta`

.. math::
    \partial_{x_i}f(x_1,\ldots x_N) \approx \frac{f(x_1,\ldots x_i + \Delta,\ldots x_N) - f(x_1,\ldots x_i,\ldots x_N)}{\Delta}

* Symbolic differentiation

    * If you put in an expression for a function, some packages will do symbolic differentiation
    * In effect, repeated applications of the chain-rule, produce-rule, etc.
    * Sometimes a good solution, if the package can handle your functions

* Automatic Differentation

    * Essentially the same as symbolic differentiation, just occurring at a different time in the compilation process 
    * Equivalent to analytical derivatives since it uses the chain-rule, etc.

We will explore AD packages in Julia rather than the alternatives

Automatic Differentiation
---------------------------

Watch the video from one of Julia's creators on `auto-differentiation <https://www.youtube.com/watch?v=vAp6nUMrKYg&feature=youtu.be>`_

To summarize here, first recall the chain rule (adapted from Wikipedia)

.. math::
    \frac{dy}{dx} = \frac{dy}{dw} \frac{dw}{dx}

Consider functions composed of calculations with fundamental operations with known analytical derivatives, such as :math:`f(x_1, x_2) = x_1 x_2 + \sin(x_1)`

To compute :math:`\frac{d f(x_1,x_2)}{d x_1}`

.. math::
    \begin{array}{l|l}
    \text{Operations to compute value} &
    \text{Operations to compute $\frac{d f(x_1,x_2)}{d x_1}$}
    \\
    \hline
    w_1 = x_1 &
    \frac{d w_1}{d x_1} = 1 \text{ (seed)}\\
    w_2 = x_2 &
    \frac{d  w_2}{d x_1} = 0 \text{ (seed)}
    \\
    w_3 = w_1 \cdot w_2 &
    \frac{d  w_3}{d x_1} = w_2 \cdot \frac{d  w_1}{d x_1} + w_1 \cdot \frac{d  w_2}{d x_1}
    \\
    w_4 = \sin w_1 &
    \frac{d  w_4}{d x_1} = \cos w_1 \cdot \frac{d  w_1}{d x_1}
    \\
    w_5 = w_3 + w_4 &
    \frac{d  w_5}{d x_1} = \frac{d  w_3}{d x_1} + \frac{d  w_4}{d x_1}
    \end{array}

Using Dual Numbers
--------------------

One way to implement this (used in Forward-mode AD) is to use `Dual Numbers <https://en.wikipedia.org/wiki/Dual_number>`_

Take a number :math:`x` and augment it with an :math:`\epsilon` such that :math:`epsilon^2 = 0`, i.e. :math:`x \to x + x' \epsilon`

All math is then done with this (mathematical, rather than Julia) tuple :math:`<x, x'>` where the :math:`x'` may be hidden from the user

With these definition, we can write a general rule for differentiation of :math:`g(x,y)` as

.. math::
    g(\left<x,x'\right>,\left<y,y'\right>) = \left<g(x,y),\partial_x g(x,y)x' + \partial_y g(x,y)y' \right>

This is is the chain rule for a total derivative

An AD library using dual numbers will concurrently calculate the function and its derivatives, repeating the chain rule until it hits a set of intrinsic rules such as

.. math::
		\begin{align*}
		x + y \to \left<x,x'\right> + \left<y,y'\right> &= \left<x + y,\underbrace{x' + y'}_{\partial(x + y) = \partial x + \partial y}\right>\\
		x y \to \left<x,x'\right> \times \left<y,y'\right> &= \left<x y,\underbrace{x'y + y'x}_{\partial(x y) = y \partial x + x \partial y y}\right>\\
		\exp(x) \to \exp(\left<x, x'\right>) &= \left<\exp(x),\underbrace{x'\exp(x)}_{\partial(\exp(x)) = \exp(x)\partial x} \right>
		\end{align*}

ForwardDiff
------------

We have already seen one of the AD packages in Julia

.. code-block:: julia

    using ForwardDiff
    h(x) = sin(x[1]) + x[1] * x[2] + sinh(x[1] * x[2]) #multivariate.
    x = [1.4 2.2]
    ForwardDiff.gradient(h,x) #uses AD, seeds from x

    #Or, can use complicated functions of many variables
    f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
    g = (x) -> ForwardDiff.gradient(f, x); #New gradient function
    x2 = rand(20)
    g(x2) #gradient at a random 20 dim point
    ForwardDiff.hessian(f,x2) #Or the hessian

We can even auto-differenitate complicated functions with embedded iterations

.. code-block:: julia

    function squareroot(x) #pretending we don't know sqrt()
        z = copy(x) # Initial starting point for Newton’s method
        while abs(z*z - x) > 1e-13
            z = z - (z*z-x)/(2z)
        end
        return z
    end
    sqrt(2.0)

.. code-block:: julia

    using ForwardDiff
    dsqrt(x) = ForwardDiff.derivative(squareroot, x)
    dsqrt(2.0)


Flux.jl
---------

Another is Flux.jl, which is a machine-learning library in Julia

AD is one of the main reasons that machine learning has become so powerful in recent years, and many machine learning libraries are effectively AD libraries  

.. code-block:: julia

    using Flux
    using Flux.Tracker
    using Flux.Tracker: update!    

    f(x) = 3x^2 + 2x + 1

    # df/dx = 6x + 2
    df(x) = Tracker.gradient(f, x)[1]

    df(2) # 14.0 (tracked)

.. code-block:: julia

    A = rand(2,2)
    f(x) = A * x
    x0 = [0.1, 2.0]
    f(x0)
    Flux.jacobian(f, x0)

As before, we can differentiate complicated functions

.. code-block:: julia

    dsquareroot(x) = Tracker.gradient(squareroot, x)    


From the docs, we can do a machine-learning approach to a linear regression

.. code-block:: julia

    W = rand(2, 5)
    b = rand(2)

    predict(x) = W*x .+ b

    function loss(x, y)
    ŷ = predict(x)
    sum((y .- ŷ).^2)
    end

    x, y = rand(5), rand(2) # Dummy data
    loss(x, y) # ~ 3

.. code-block:: julia


    W = param(W)
    b = param(b)

    gs = Tracker.gradient(() -> loss(x, y), Params([W, b]))

    Δ = gs[W]

    # Update the parameter and reset the gradient
    update!(W, -0.1Δ)

    loss(x, y) # ~ 2.5

JuMP 
--------

.. code-block:: julia

    using JuMP, Ipopt
    # Solve constrained optimization 
    # max( x[1] + x[2] )
    # st sqrt(x[1]^2 + x[2]^2) <= 1

    #Can even auto-differenitate complicated functions with embedded iterations
    function squareroot(x) #pretending we don't know sqrt()
        z = x # Initial starting point for Newton’s method
        while abs(z*z - x) > 1e-13
            z = z - (z*z-x)/(2z)
        end
        return z
    end
    m = Model(solver = IpoptSolver())
    JuMP.register(m,:squareroot, 1, squareroot, autodiff=true) # For user defined complicated functions


    @variable(m, x[1:2], start=0.5)
    @objective(m, Max, sum(x))
    @NLconstraint(m, squareroot(x[1]^2+x[2]^2) <= 1)
    solve(m)

.. code-block:: julia

    using JuMP,Ipopt
    m = Model(solver = IpoptSolver())
    @variable(m, x, start = 0.0)
    @variable(m, y, start = 0.0)

    @NLobjective(m, Min, (1-x)^2 + 100(y-x^2)^2)

    solve(m)
    println("x = ", getvalue(x), " y = ", getvalue(y))

    # adding a (linear) constraint
    @constraint(m, x + y == 10)
    solve(m)
    println("x = ", getvalue(x), " y = ", getvalue(y))

Roots
=========================================


Let's look briefly at the optimization and root finding algorithms


Roots
-----------------

A root of a real function :math:`f` on :math:`[a,b]` is an :math:`x \in [a, b]` such that :math:`f(x)=0`

For example, if we plot the function

.. math::
    :label: root_f

    f(x) = \sin(4 (x - 1/4)) + x + x^{20} - 1


with :math:`x \in [0,1]` we get

.. _root_fig:


The unique root is approximately 0.408

The `Roots <https://github.com/JuliaLang/Roots.jl>`_ package offers the ``fzero()`` to find roots

.. code-block:: julia

    using Roots

.. code-block:: julia

    f(x) = sin(4 * (x - 1/4)) + x + x^20 - 1

.. code-block:: julia

    fzero(f, 0, 1)


Optimization
---------------------

For constrained, univariate minimization a useful option is ``maximize()`` from the
`Optim <https://github.com/JuliaOpt/Optim.jl>`_ package

This function defaults to a robust hybrid optimization routine called Brent's method

.. code-block:: julia

    using Optim

    maximize(x -> x^2, -1.0, 1.0)


For other optimization routines, including least squares and multivariate optimization, see `the documentation <https://github.com/JuliaOpt/Optim.jl/blob/master/README.md>`_

A number of alternative packages for optimization can be found at `JuliaOpt <http://www.juliaopt.org/>`_
