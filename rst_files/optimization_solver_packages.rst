.. _optimization_solver_packages:

.. include:: /_static/includes/lecture_howto_jl.raw

***************************************************
Solvers, Optimizers, and Automatic Differentiation
***************************************************

.. contents:: :depth: 2

Overview
============

In this lecture we introduce a few of the Julia libraries that we've found particularly useful for quantitative work in economics

Setup
------------------

.. literalinclude:: /_static/includes/alldeps.jl

Introduction to Automatic Differentiation
=============================================

Automatic differentiation (sometimes called algorithmic differentiation) is a crucial way to increase the performance of both estimation and solution methods

There are essentially three four ways to calculate the gradient or jacobian on a computer

* Calculation by hand

    * Where possible, you can calculate the derivative on "pen-and-paper" and potentially simplify the expression
    * Sometimes, though not always, the most accurate and fastest option if there are algebraic simplifications
    * The algebra is error prone for non-trivial setups

* Finite differences

    * Evaluate the function at least ``N`` times to get the gradient, jacobians are even worse
    * Large :math:`\Delta` is numerically stable but inaccurate, too small of :math:`\Delta` is numerically unstable but more accurate
    * Avoid if you can, and use packages (e.g. `DiffEqDiffTools.jl <https://github.com/JuliaDiffEq/DiffEqDiffTools.jl>`_ ) to get a good choice of :math:`\Delta`

.. math::
    \partial_{x_i}f(x_1,\ldots x_N) \approx \frac{f(x_1,\ldots x_i + \Delta,\ldots x_N) - f(x_1,\ldots x_i,\ldots x_N)}{\Delta}

* Symbolic differentiation

    * If you put in an expression for a function, some packages will do symbolic differentiation
    * In effect, repeated applications of the chain-rule, produce-rule, etc.
    * Sometimes a good solution, if the package can handle your functions

* Automatic Differentiation

    * Essentially the same as symbolic differentiation, just occurring at a different time in the compilation process 
    * Equivalent to analytical derivatives since it uses the chain-rule, etc.

We will explore AD packages in Julia rather than the alternatives

Automatic Differentiation
---------------------------

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

Take a number :math:`x` and augment it with an infinitesimal :math:`\epsilon` such that :math:`\epsilon^2 = 0`, i.e. :math:`x \to x + x' \epsilon`

All math is then done with this (mathematical, rather than Julia) tuple :math:`(x, x')` where the :math:`x'` may be hidden from the user

With these definition, we can write a general rule for differentiation of :math:`g(x,y)` as

.. math::
    g(\left(x,x'\right),\left(y,y'\right)) = \left(g(x,y),\partial_x g(x,y)x' + \partial_y g(x,y)y' \right)

This calculation is simply the chain rule for the total derivative

An AD library using dual numbers will concurrently calculate the function and its derivatives, repeating the chain rule until it hits a set of intrinsic rules such as

.. math::
		\begin{align*}
		x + y \to \left(x,x'\right) + \left(y,y'\right) &= \left(x + y,\underbrace{x' + y'}_{\partial(x + y) = \partial x + \partial y}\right)\\
		x y \to \left(x,x'\right) \times \left(y,y'\right) &= \left(x y,\underbrace{x'y + y'x}_{\partial(x y) = y \partial x + x \partial y y}\right)\\
		\exp(x) \to \exp(\left(x, x'\right)) &= \left(\exp(x),\underbrace{x'\exp(x)}_{\partial(\exp(x)) = \exp(x)\partial x} \right)
		\end{align*}

ForwardDiff.jl
-----------------

We have already seen one of the AD packages in Julia

.. code-block:: julia

    using ForwardDiff
    h(x) = sin(x[1]) + x[1] * x[2] + sinh(x[1] * x[2]) # multivariate.
    x = [1.4 2.2]
    @show ForwardDiff.gradient(h,x) # use AD, seeds from x

    #Or, can use complicated functions of many variables
    f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x)
    g = (x) -> ForwardDiff.gradient(f, x); # g() is now the gradient
    @show g(rand(20)); # gradient at a random point
    # ForwardDiff.hessian(f,x2) # or the hessian

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

AD is one of the main reasons that machine learning has become so powerful in recent years, and is an essential component of any ML package

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


From the documentation, we can do a machine-learning approach to a linear regression

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


Optimization
===============

There are a large number of packages intended to be used for optimization in Julia

Part of the reason for the diversity of options is that Julia makes it possible to efficiently implement a large number of variations on optimization routines

The other reason is that different types of optimization problems require different algorithms

.. TODO: need to describe different types of optimization problems and suggested packages at some point


Optim.jl
---------------------

A good pure-Julia solution for the (unconstrained or box-bounded) optimization of univariate and multivariate function is `Optim <https://github.com/JuliaNLSolvers/Optim.jl>`_ package

By default, the algorithms in Optim target minimization rather than maximization, so if a function is called "optimize" it will mean minimization

Univariate Functions on Bounded Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Univariate optimization <http://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#minimizing-a-univariate-function-on-a-bounded-interval>`_ defaults to a robust hybrid optimization routine called Brent's method

.. code-block:: julia

    using Optim
    using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

    result = optimize(x-> x^2, -2.0, 1.0)


Always check if the results converged, and throw errors otherwise

.. code-block:: julia

    converged(result) || error("Failed to converge in $(iterations(result)) iterations")
    xmin = result.minimizer
    result.minimum


Or to maximize

.. code-block:: julia

    f(x) = -x^2
    result = maximize(f, -2.0, 1.0)
    converged(result) || error("Failed to converge in $(iterations(result)) iterations")
    xmin = maximizer(result)
    fmax = maximum(result)


**Note:** There are a few inconsistencies with extracting  ``optimize`` vs. ``maximize`` results, as shown above

Unconstrained Multivariate Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a variety of `algorithms and options <http://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#_top>`_ for multivariate optimization

From the documentation, the simplest version is

.. code-block:: julia

    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    x_iv = [0.0, 0.0]
    results = optimize(f, x_iv) # i.e. optimize(f, x_iv, NelderMead())

The default algorithm in ``NelderMead``, which is derivative-free and hence requires many function evaluations

To change the algorithm type to `L-BFGS <http://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/>`_

.. code-block:: julia

    results = optimize(f, x_iv, LBFGS())
    println("minimum = $(results.minimum) with argmin = $(results.minimizer) in $(results.iterations) iterations")    

Note that this has fewer iterations

However, since no derivative was given, it had to use `finite-differences <https://github.com/JuliaDiffEq/DiffEqDiffTools.jl>`_ to approximate the gradient of ``f(x)``


However, since most of the algorithms require derivatives, you will often want to use auto differentiation or pass analytical gradients if possible

.. code-block:: julia

    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    x_iv = [0.0, 0.0]
    results = optimize(f, x_iv, LBFGS(), autodiff=:forward) # i.e. use ForwardDiff.jl
    println("minimum = $(results.minimum) with argmin = $(results.minimizer) in $(results.iterations) iterations")    

Note that we did not need to use ``ForwardDiff.jl`` directly, as long as our ``f(x)`` function was written to be generic (see the `tips and trick <generic_tips_tricks>`_ )

Alternatively, with an analytical gradient

.. code-block:: julia

    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    x_iv = [0.0, 0.0]
    function g!(G, x)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end

    results = optimize(f, g!, x0, LBFGS()) # or ConjugateGradient()
    println("minimum = $(results.minimum) with argmin = $(results.minimizer) in $(results.iterations) iterations")    

For derivative-free methods, you can change the algorithm--and have no need to provide a gradient

.. code-block:: julia

    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    x_iv = [0.0, 0.0]
    results = optimize(f, x_iv, SimulatedAnnealing()) # or ParticleSwarm() or NelderMead()

However, you will note that this did not converge, as stochastic methods typically require many more iterations as a tradeoff for their global-convergence properties


See the `maximum likelihood <http://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/maxlikenlm/>`_ example and the accompanying `jupyter notebook <https://nbviewer.jupyter.org/github/JuliaNLSolvers/Optim.jl/blob/gh-pages/v0.15.3/examples/generated/maxlikenlm.ipynb>`_

JuMP.jl
--------

The `JuMP.jl <https://github.com/JuliaOpt/JuMP.jl>`_ package is an ambitious implementation of a modelling language for optimization problems in Julia

In that sense, it is more like an AMPL (or Pyomo) built on top of the Julia language with macros, and able to use a variety of different commerical and open-source solvers

If you have a linear, quadratic, conic, mixed-integer linear, etc. problem then this will likely be the ideal "meta-package" for calling various solvers

For nonlinear problems, the modelling language may make things difficult for complicated functions (as it is not designed to be used as a general-purpose non-linear optimizer) 

See the `quickstart guide <http://www.juliaopt.org/JuMP.jl/0.18/quickstart.html>`_ for more details on all of the options

The following is an example of 

.. code-block:: julia

    using JuMP, Ipopt
    # solve
    # max( x[1] + x[2] )
    # st sqrt(x[1]^2 + x[2]^2) <= 1

    function squareroot(x) # pretending we don't know sqrt()
        z = x # Initial starting point for Newton’s method
        while abs(z*z - x) > 1e-13
            z = z - (z*z-x)/(2z)
        end
        return z
    end
    m = Model(solver = IpoptSolver())
    JuMP.register(m,:squareroot, 1, squareroot, autodiff=true) # need to register user defined functions for AD

    @variable(m, x[1:2], start=0.5) # start is the initial condition
    @objective(m, Max, sum(x))
    @NLconstraint(m, squareroot(x[1]^2+x[2]^2) <= 1)
    solve(m)


And this is an example of a quadratic objective

.. code-block:: julia

    # solve
    # min (1-x)^2 + 100(y-x^2)^2)
    # st x + y >= 10

    using JuMP,Ipopt
    m = Model(solver = IpoptSolver(print_level=0)) # settings for the solver, e.g. suppress output
    @variable(m, x, start = 0.0)
    @variable(m, y, start = 0.0)

    @NLobjective(m, Min, (1-x)^2 + 100(y-x^2)^2)

    solve(m)
    println("x = ", getvalue(x), " y = ", getvalue(y))

    # adding a (linear) constraint
    @constraint(m, x + y == 10)
    solve(m)
    println("x = ", getvalue(x), " y = ", getvalue(y))


BlackBoxOptim.jl
---------------------

Another package for doing global optimization without derivatives is `BlackBoxOptim.jl <https://github.com/robertfeldt/BlackBoxOptim.jl>`_

To see an example from the documentation, 

.. code-block:: julia

    using BlackBoxOptim

    function rosenbrock2d(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end

    results = bboptimize(rosenbrock2d; SearchRange = (-5.0, 5.0), NumDimensions = 2);


An example for `parallel execution <https://github.com/robertfeldt/BlackBoxOptim.jl/blob/master/examples/rosenbrock_parallel.jl>`_ of the objective is provided

Systems of Equations
===============================

Roots.jl
------------------------------------

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

NLsolve.jl
------------------

The package `NLsolve.jl <https://github.com/JuliaNLSolvers/NLsolve.jl/>`_ provides functions to solve for multivariate systems of equations and fixed-points

From the documentation, to solve for a system of equations without providing a jacobian

.. code-block:: julia

    using NLsolve

    f(x) = [(x[1]+3)*(x[2]^3-7)+18
            sin(x[2]*exp(x[1])-1)] # returns an array

    results = nlsolve(f, [ 0.1; 1.2], inplace=false) # the inplace will not be necessary in future versions

In the above case, the algorithm used finite-differences to calculate the jacobian

Alternatively, if ``f(x)`` is written generically, you can use auto-differentiation with a single setting

.. code-block:: julia

    results = nlsolve(f, [ 0.1; 1.2], autodiff=:forward, inplace=false)
    
    println("converged=$(NLsolve.converged(results)) at root=$(results.zero) in $(results.iterations) iterations and $(results.f_calls) function calls")


Providing a function with operates in-place (i.e. modifying an argument) may help performance for large systems of equations (and hurt it for small ones)

.. code-block:: julia

    function f!(F, x) # modifies the first argument
        F[1] = (x[1]+3)*(x[2]^3-7)+18
        F[2] = sin(x[2]*exp(x[1])-1)
    end

    results = nlsolve(f!, [ 0.1; 1.2], autodiff=:forward)

    println("converged=$(NLsolve.converged(results)) at root=$(results.zero) in $(results.iterations) iterations and $(results.f_calls) function calls")


Exercises
=============


Exercise 1
------------

Watch `this video <https://www.youtube.com/watch?v=vAp6nUMrKYg&feature=youtu.be>`_ from one of Julia's creators on automatic differentiation