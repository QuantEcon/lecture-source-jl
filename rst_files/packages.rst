.. _packages:

.. include:: /_static/includes/lecture_howto_jl.raw

**********************
Useful Libraries
**********************

.. contents:: :depth: 2

Overview
============

While Julia lacks the massive scientific ecosystem of Python, it has successfully attracted a small army of enthusiastic and talented developers

As a result, its package system is moving towards a critical mass of useful, well written libraries

In addition, a major advantage of Julia libraries is that, because Julia itself is sufficiently fast, there is less need to mix in low level languages like C and Fortran

As a result, most Julia libraries are written exclusively in Julia

Not only does this make the libraries more portable, it makes them much easier to dive into, read, learn from and modify

In this lecture we introduce a few of the Julia libraries that we've found particularly useful for quantitative work in economics

Setup
------------------

.. literalinclude:: /_static/includes/deps.jl

Distributions
=====================

Functions for manipulating probability distributions and generating random
variables are supplied by the excellent `Distributions.jl <https://github.com/JuliaStats/Distributions.jl>`_ package

We'll restrict ourselves to a few simple examples (the package itself has `detailed documentation <https://juliastats.github.io/Distributions.jl/latest/index.html>`_)

* ``d = Normal(m, s)`` creates a normal distribution with mean :math:`m` and standard deviation :math:`s`

    * defaults are ``m = 0`` and ``s = 1``

* ``d = Uniform(a, b)`` creates a uniform distribution on interval :math:`[a, b]`

    * defaults are ``a = 0`` and ``b = 1``

* ``d = Binomial(n, p)`` creates a binomial over :math:`n` trials with success probability :math:`p`

* defaults are ``n = 1`` and ``p = 0.5``

Distributions.jl defines various methods for acting on these instances in order to obtain

* random draws

* evaluations of pdfs (densities), cdfs (distribution functions), quantiles, etc.

* mean, variance, kurtosis, etc.

For example,

* To generate ``k`` draws from the instance ``d`` use ``rand(d, k)``

* To obtain the mean of the distribution use ``mean(d)``

* To evaluate the probability density function of ``d`` at ``x`` use ``pdf(d, x)``

Further details on the interface can be found `here <https://juliastats.github.io/Distributions.jl/latest/univariate.html#Common-Interface-1>`__

Several multivariate distributions are also implemented


.. _df:

Working with Data
========================

A useful package for working with data is `DataFrames <https://github.com/JuliaStats/DataFrames.jl>`_

The most important data type provided is a ``DataFrame``, a two dimensional array for storing heterogeneous data

Although data can be heterogeneous within a ``DataFrame``, the contents of the columns must be homogeneous

This is analogous to a ``data.frame`` in R, a ``DataFrame`` in Pandas (Python) or, more loosely, a spreadsheet in Excel

Here's an example

Activate the project environment, ensuring that ``Project.toml`` and ``Manifest.toml`` are in the same location as your notebook

.. code-block:: julia

    using Pkg; Pkg.activate(@__DIR__); #activate environment in the notebook's location

.. code-block:: julia

    using DataFrames

.. code-block:: julia

    commodities = ["crude", "gas", "gold", "silver"]

.. code-block:: julia

    last_price = [4.2, 11.3, 12.1, missing]

.. code-block:: julia

    df = DataFrame(commod = commodities, price = last_price)


Columns of the DataFrame can be accessed by name


.. code-block:: julia

    df[:price]


.. code-block:: julia

    df[:commod]


The DataFrames package provides a number of methods for acting on DataFrames

A simple one is ``describe()``

.. code-block:: julia

    describe(df)


There are also functions for splitting, merging and other data munging
operations

Data can be read from and written to CSV files using the CSV package

.. code-block:: julia
    :class: no-execute

    using CSV
    df = CSV.read("data_file.csv")
    CSV.write("data_file.csv", df)

Other packages for working with data can be found at `JuliaStats <https://github.com/JuliaStats>`_ and `JuliaQuant <https://github.com/JuliaQuant>`_


Interpolation
=============================

In economics we often wish to interpolate discrete data (i.e., build continuous functions that join discrete sequences of points)

We also need such representations to be fast and efficient

The package we usually turn to for this purpose is `Interpolations.jl <https://github.com/JuliaMath/Interpolations.jl>`_

One downside of Interpolations.jl is that the code to set up simple interpolation objects is relatively verbose

The upside is that the routines have excellent performance

The package is also well written and well maintained

Another alternative, if using univariate linear interpolation, is `LinInterp` from `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`_

As we show below, the syntax for this function is much simpler


Univariate Interpolation
---------------------------

Let's start with the univariate case

We begin by creating some data points, using a sine function

.. code-block:: julia

    using Interpolations
    using Plots
    gr(fmt=:png)

    x = -7:7             # x points, coase grid
    y = sin.(x)          # corresponding y points

    xf = -7:0.1:7        # fine grid
    plot(xf, sin.(xf), label = "sine function")
    scatter!(x, y, label = "sampled data", markersize = 4)


We can implement linear interpolation easily using QuantEcon's `LinInterp`


.. code-block:: julia

    using QuantEcon

    li = LinInterp(x, y)        # create LinInterp object
    li(0.3)                     # evaluate at a single point
    y_linear_qe = li.(xf)       # evaluate at multiple points

    plot(xf, y_linear_qe, label = "linear")
    scatter!(x, y, label = "sampled data", markersize = 4)


The syntax is simple and the code is efficient, but for other forms of
interpolation we need a more sophisticated set of routines


As an example, let's employ `Interpolations.jl` to interpolate the sampled data points using piecewise constant, piecewise linear and cubic interpolation


.. code-block:: julia

    itp_const = scale(interpolate(y, BSpline(Constant()), OnGrid()), x)
    itp_linear = scale(interpolate(y, BSpline(Linear()), OnGrid()), x)
    itp_cubic = scale(interpolate(y, BSpline(Cubic(Line())), OnGrid()), x)


When we want to evaluate them at points in their domain (i.e., between
``min(x)`` and ``max(x)``) we can do so as follows


.. code-block:: julia

    itp_cubic[0.3]


Note the use of square brackets, rather than parentheses!

Let's plot these objects created above

.. code-block:: julia

    xf = -7:0.1:7
    y_const = [ itp_const[x] for x ∈ xf ]
    y_linear = [ itp_linear[x] for x ∈ xf ]
    y_cubic = [ itp_cubic[x] for x ∈ xf ]

    plot(xf, [y_const y_linear y_cubic], label = ["constant" "linear" "cubic"])
    scatter!(x, y, label = "sampled data", markersize = 4)



Univariate with Irregular Grid
---------------------------------

The `LinInterp` from `QuantEcon.jl <https://github.com/QuantEcon/QuantEcon.jl>`_ works the same whether or not the grid is regular (i.e., evenly spaced)

The `Interpolations.jl` code is a bit more picky

Here's an example of the latter with an irregular grid

.. code-block:: julia

    x = log.(range(1,  exp(4), length = 10)) .+ 1  # Uneven grid
    y = log.(x)                            # Corresponding y points

    itp_const = interpolate((x, ), y, Gridded(Constant()))
    itp_linear = interpolate((x, ), y, Gridded(Linear()))

    xf = log.(range(1,  exp(4), length = 100)) .+ 1
    y_const = [ itp_const[x] for x ∈ xf ]
    y_linear = [ itp_linear[x] for x ∈ xf ]
    y_true = [ log(x) for x ∈ xf ]

    labels = ["piecewise constant" "linear" "true function"]
    plot(xf, [y_const y_linear y_true], label = labels)
    scatter!(x, y, label = "sampled data", markersize = 4, size = (800, 400))


Multivariate Interpolation
-----------------------------

We can also interpolate in higher dimensions

The following example gives one illustration

.. code-block:: julia

    let
        n = 5
        x = range(-3,  3, length = n)
        y = copy(x)

        z = zeros(n, n)
        f(x, y) = cos(x^2 + y^2) / (1 + x^2 + y^2)
        for i ∈ 1:n
            for j ∈ 1:n
                z[j, i] = f(x[i], y[j])
            end
        end

        itp = interpolate((x, y), z, Gridded(Linear()));

        nf = 50
        xf = range(-3,  3, length = nf)
        yf = copy(xf)

        zf = zeros(nf, nf)
        ztrue = similar(zf)
        for i ∈ 1:nf
            for j ∈ 1:nf
                zf[j, i] = itp[xf[i], yf[j]]
                ztrue[j, i] = f(xf[i], yf[j])
            end
        end

        grid = gridmake(x, y)
        z = reshape(z, n * n, 1)

        surface(xf, yf, zf', color = :greens, falpha = 0.7, cbar = false)
        surface!(xf, yf, ztrue', fcolor = :blues, falpha = 0.25, cbar = false)
        scatter!(grid[:, 1], grid[:, 2], vec(z), legend = :none, color = :black,
                 markersize = 4)
    end

The original function is in blue, while the linear interpolant is shown in green

Optimization, Roots and Fixed Points
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


Other Topics
=================

Numerical Integration
-----------------------

The `QuadGK <https://github.com/JuliaMath/QuadGK.jl>`_ library contains a function called ``quadgk()`` that performs
Gaussian quadrature

.. code-block:: julia

    using QuadGK
    quadgk(cos, -2π, 2π)


This is an adaptive Gauss-Kronrod integration technique that's relatively accurate for smooth functions

However, its adaptive implementation makes it slow and not well suited to inner loops

For this kind of integration you can use the quadrature routines from QuantEcon

.. code-block:: julia

    nodes, weights = qnwlege(65, -2π, 2π);
    integral = do_quad(x -> cos.(x), nodes, weights)


Let's time the two implementations

.. code-block:: julia

    using BenchmarkTools
    @btime quadgk(x -> cos.(x), -2π, 2π)
    @btime do_quad(x -> cos.(x), $nodes, $weights)


We get similar accuracy with a speed up factor approaching three orders of magnitude

More numerical integration (and differentiation) routines can be found in the
package `Calculus <https://github.com/JuliaMath/Calculus.jl>`_


Linear Algebra
-----------------

The standard library contains many useful routines for linear algebra, in
addition to standard functions such as ``det()``, ``inv()``, ``factorize()``, etc.

Routines are available for

* Cholesky factorization

* LU decomposition

* Singular value decomposition,

* Schur factorization, etc.

See `here <https://docs.julialang.org/en/stable/manual/linear-algebra/>`__ for further details


Further Reading
===================

The full set of libraries available under the Julia packaging system can be browsed at `pkg.julialang.org <http://pkg.julialang.org/>`_
