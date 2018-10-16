.. _plots:

.. include:: /_static/includes/lecture_howto_jl.raw

**********************
Plotting in Julia
**********************

.. contents:: :depth: 2


Overview
============

Activate the ``QuantEconLecturePackages`` project environment and package versions

.. code-block:: julia 

    using InstantiateFromURL
    activate_github("QuantEcon/QuantEconLecturePackages")
    using LinearAlgebra, Statistics, Compat

Since it's inception, plotting in Julia has been a mix of happiness and frustration

Some initially promising libraries have stagnated, or failed to keep up with user needs

New packages have appeared to compete with them, but not all are fully featured

The good news is that the Julia community now has several very good options for plotting

In this lecture we'll try to save you some of our pain by focusing on what we believe are currently the best libraries

First we look at two high quality plotting packages that have proved useful to us in a range of applications

After that we turn to a relative newcomer called `Plots.jl <https://juliaplots.github.io/>`__ 

The latter package takes a different -- and intriguing -- approach that combines and exploits the strengths of several existing plotting libraries

Below we assume that 

* you've already read through :doc:`our getting started lecture <getting_started>` 

* you are working in a Jupyter notebook, as described :ref:`here <jl_jupyter>`


How to Read this Lecture
-------------------------

If you want to get started quickly with relatively simple plots, you can skip straight 
to the :ref:`section on Plots.jl <plotsjl>`

If you want a deeper understanding and more flexibility, continue from the next section and read on

PyPlot
========================

Let's look at `PyPlot <https://github.com/stevengj/PyPlot.jl>`_ first

PyPlot is a Julia front end to the excellent Python plotting library `Matplotlib <http://matplotlib.org/>`_



Installing PyPlot
----------------------

One disadvantage of PyPlot is that it not only requires Python but also much of the scientific Python stack

Fortunately, installation of the latter has been greatly simplified by the excellent Anaconda Python distribution

Moreover, the tools that come with Anaconda (such as Jupyter) are too good to miss out on 

So please go ahead and :ref:`install Anaconda <install_anaconda>` if you haven't yet

Next, start up Julia and type ``Pkg.add("PyPlot")``


Usage
--------

There are two different interfaces to Matplotlib and hence to PyPlot

Let's look at them in turn

The Procedural API
------------------------

Matplotlib has a straightforward plotting API that essentially replicates the plotting routines in MATLAB

These plotting routines can be expressed in Julia with almost identical syntax

Here's an example

.. code-block:: julia

    using PyPlot
    x = linspace(0, 10, 200)
    y = sin.(x)
    plot(x, y, "b-", linewidth=2)


3D Plots
^^^^^^^^^^

Here's an example of how to create a 3D plot 

.. code-block:: julia
   
   using QuantEcon: meshgrid
   
   n = 50
   x = linspace(-3, 3, n)
   y = x
   
   z = Array{Float64}(n, n)
   f(x, y) = cos(x^2 + y^2) / (1 + x^2 + y^2)
   for i in 1:n
       for j in 1:n
           z[j, i] = f(x[i], y[j])
       end
   end
   
   xgrid, ygrid = meshgrid(x, y)
   surf(xgrid, ygrid, z', cmap=ColorMap("jet"), alpha=0.7)
   zlim(-0.5, 1.0)



The Object Oriented API
------------------------

Matplotlib also has a more powerful and expressive object oriented API 

Because Julia isn't object oriented in the same sense as Python, 
the syntax required to access this interface via PyPlot is a little awkward

Here's an example

.. code-block:: julia

    x = linspace(0, 10, 200)
    y = sin.(x)
    fig, ax = subplots()
    ax[:plot](x, y, "b-", linewidth=2)
    

The resulting figure is the same

Here we get no particular benefit from switching APIs, while introducing a less attractive syntax

However, as plots get more complex, the more explicit syntax will give us greater control

Here's a similar plot with a bit more customization

.. code-block:: julia

    x = linspace(0, 10, 200)
    y = sin.(x)
    fig, ax = subplots()
    ax[:plot](x, y, "r-", linewidth=2, label="sine function", alpha=0.6)
    ax[:legend](loc="upper center")
    

The resulting figure has a legend at the top center


We can render the legend in LaTeX by changing the ``ax[:plot]`` line to

.. code-block:: julia

    x = linspace(0, 10, 200)
    y = sin.(x)
    fig, ax = subplots()
    ax[:plot](x, y, "r-", linewidth=2, label=L"$y = \sin(x)$", alpha=0.6)
    ax[:legend](loc="upper center")



Note the ``L`` in front of the string to indicate LaTeX mark up

.. _mpoa:

Multiple Plots on One Axis
-----------------------------

Here's another example, which helps illustrate how to put multiple plots on one figure

We use `Distributions.jl` to get the values of the densities given a randomly generated mean and standard deviation

.. code-block:: julia

    using Distributions
    
    u = Uniform()
    
    fig, ax = subplots()
    x = linspace(-4, 4, 150)
    for i in 1:3
        # == Compute normal pdf from randomly generated mean and std == #
        m, s = rand(u) * 2 - 1, rand(u) + 1
        d = Normal(m, s)
        y = pdf.(d, x)
        # == Plot current pdf == #
        ax[:plot](x, y, linewidth=2, alpha=0.6, label="draw $i")
    end
    ax[:legend]()


.. _pyplot_sub:

Subplots
^^^^^^^^^^^^^^^^^^^

A figure containing ``n`` rows and ``m`` columns of subplots can be created by
the call

.. code-block:: julia
    :class: no-execute

    fig, axes = subplots(num_rows, num_cols)

Here's an example that generates 6 normal distributions, takes 100 draws from each, and plots each of the resulting histograms 

.. code-block:: julia

    u = Uniform()
    num_rows, num_cols = 2, 3
    fig, axes = subplots(num_rows, num_cols, figsize=(16,6))
    subplot_num = 0
    
    for i in 1:num_rows
        for j in 1:num_cols
            ax = axes[i, j]
            subplot_num += 1
            # == Generate a normal sample with random mean and std == #
            m, s = rand(u) * 2 - 1, rand(u) + 1
            d = Normal(m, s)
            x = rand(d, 100)
            # == Histogram the sample == #
            ax[:hist](x, alpha=0.6, bins=20)
            ax[:set_title]("histogram $subplot_num")
            ax[:set_xticks]([-4, 0, 4])
            ax[:set_yticks]([])
        end
    end



PlotlyJS
============

Now let's turn to another plotting package --- a promising new library called `PlotlyJS <https://github.com/spencerlyon2/PlotlyJS.jl>`_, authored by `Spencer Lyon <https://github.com/sglyon>`_

PlotlyJS is a Julia interface to the `plotly.js visualization library <https://plot.ly/javascript/>`_

It can be installed by typing ``Pkg.add("PlotlyJS")`` from within Julia

It has several advantages, one of which is beautiful interactive plots


While we won't treat the interface in great detail, we will frequently use PlotlyJS as a backend for Plots.jl

(More on this below)


Examples
-------------

Let's look at some simple examples

Here's a version of the sine function plot you saw above


.. code-block:: julia

    import PlotlyJS
    x = linspace(0, 10, 200)
    y = sin.(x)
    # specify which module scatter belongs to since both have scatter
    PlotlyJS.plot(PlotlyJS.scatter(x=x, y=y, marker_color="blue", line_width=2)) 



Here's a replication of the :ref:`figure with multiple Gaussian densities <mpoa>`

.. code-block:: julia

    traces = PlotlyJS.GenericTrace[]
    u = Uniform()
    
    x = linspace(-4, 4, 150)
    for i in 1:3
        # == Compute normal pdf from randomly generated mean and std == #
        m, s = rand(u) * 2 - 1, rand(u) + 1
        d = Normal(m, s)
        y = pdf.(d, x)
        trace = PlotlyJS.scatter(x=x, y=y, name="draw $i")
        push!(traces, trace)
    end
    
    PlotlyJS.plot(traces, PlotlyJS.Layout())


The output looks like this (modulo randomness):




.. _plotsjl:

Plots.jl
=====================

`Plots.jl <https://github.com/tbreloff/Plots.jl>`__ is another relative newcomer to the Julia plotting scene, authored by `Tom Breloff <https://github.com/tbreloff>`_

The approach of Plots.jl is to 

#. provide a "frontend" plotting language

#. render the plots by using one of several existing plotting libraries as "backends" 

In other words, Plots.jl plotting commands are translated internally to commands understood by a selected plotting library

Underlying libraries, or backends, can be swapped very easily

This is neat because each backend has a different look, as well as different capabilities

Also, Julia being Julia, it's quite possible that a given backend won't install or function on your machine at a given point in time

With Plots.jl, you can just change to another one



Simple Examples
---------------

We produced some simple plots using Plots.jl back in :doc:`our introductory Julia lecture <julia_by_example>`

Here's another simple one:

.. code-block:: julia

    import Plots
    x = linspace(0, 10, 200)
    y = sin.(x)
    Plots.plot(x, y, color=:blue, linewidth=2, label="sine")



No backend was specified in the preceding code, and in this case it defaulted to Plots.jl

We can make this explicit by adding one extra line

.. code-block:: julia

    Plots.pyplot()   # specify backend
    x = linspace(0, 10, 200)
    y = sin.(x)
    Plots.plot(x, y, color=:blue, linewidth=2, label="sine")

To switch your backend to PlotlyJS, change ``pyplot()`` to ``plotlyjs()``

Your figure should now look more like the plots produced by PlotlyJS

Here's a slightly more complex plot using Plots.jl with PyPlot backend

.. code-block:: julia

    using LaTeXStrings                  # Install this package
    Plots.pyplot()
    x = linspace(0, 10, 100)
    Plots.plot(x,
               sin,
               color=:red,
               lw=2,
               yticks=-1:1:1,
               title="sine function",
               label=L"$y = \sin(x)$",  # L for LaTeX string
               alpha=0.6)




Use ``legend=:none`` if you want no legend on the plot

Notice that in the preceding code example, the second argument to `plot()` is a function rather than an array of data points

This is valid syntax, as is

.. code-block:: julia

    Plots.plot(sin, 0, 10)   # Plot the sine function from 0 to 10
    

Plots.jl accommodates these useful variations in syntax by exploiting multiple dispatch 


Multiple Plots on One Axis
-----------------------------

Next, let's replicate the :ref:`figure with multiple Gaussian densities <mpoa>` 


.. code-block:: julia

    Plots.plotlyjs()
    
    x = linspace(-4, 4, 150)
    y_vals = Array{Vector}(3)
    labels = Array{String}(1, 3)
    for i = 1:3
        m, s = 2*(rand() - 0.5), rand() + 1
        d = Normal(m, s)
        y_vals[i] = pdf.(d, x)
        labels[i] = string("mu = ", round(m, 2))
    end
    
    Plots.plot(x, y_vals, linewidth=2, alpha=0.6, label=labels)

Also, when you have multiple y-series, `Plots.jl` can accept one x-values vector and apply it to each y-series




Subplots
-----------------------------

Let's replicate the subplots figure :ref:`shown above <pyplot_sub>`

.. code-block:: julia

    Plots.pyplot()
    
    draws = Array{Vector}(6)
    titles = Array{String}(1, 6)
    for i = 1:6
        m, s = 2*(rand() - 0.5), rand() + 1
        d = Normal(m, s)
        draws[i] = rand(d, 100)
        t = string(L"$\mu = $", round(m, 2), L", $\sigma = $", round(s, 2))
        titles[i] = t
    end
    
    Plots.histogram(draws,
                    layout=6,
                    title=titles,
                    legend=:none,
                    titlefont=Plots.font(9),
                    bins=20)


Notice that the font and bins settings get applied to each subplot


When you want to pass individual arguments to subplots, you can use a row vector of arguments

* For example, in the preceding code, ``titles'`` is a `1 x 6` row vector

Here's another example of this, with a row vector of different colors for the
histograms

.. code-block:: julia

    Plots.pyplot()

    draws = Array{Vector}(6)
    titles = Array{String}(1, 6)
    for i = 1:6
        m, s = 2*(rand() - 0.5), rand() + 1
        d = Normal(m, s)
        draws[i] = rand(d, 100)
        t = string(L"$\mu = $", round(m, 2), L", $\sigma = $", round(s, 2))
        titles[i] = t
    end

    Plots.histogram(draws, 
                    layout=6, 
                    title=titles, 
                    legend=:none, 
                    titlefont=Plots.font(9), 
                    color=[:red :blue :yellow :green :black :purple],
                    bins=20)


The result is a bit garish but hopefully the message is clear



3D Plots
---------

Here's a sample 3D plot

.. code-block:: julia

    Plots.plotlyjs()
    
    n = 50
    x = linspace(-3, 3, n)
    y = x
    
    z = Array{Float64}(n, n)
    ff(x, y) = cos(x^2 + y^2) / (1 + x^2 + y^2)
    for i in 1:n
        for j in 1:n
            z[j, i] = ff(x[i], y[j])
        end
    end
    
    Plots.surface(x, y, z')



Further Reading
---------------

Hopefully this tutorial has given you some ideas on how to get started with Plots.jl

We'll see more examples of this package in action through the lectures

Additional information can be found in the `official documentation <https://juliaplots.github.io/>`_

Exercises
===========


Exercise 1
----------

The identity function :math:`f(x) = x` is approximated on the nonnegative numbers
:math:`[0, \infty)` with increasing degrees of precision by the sequence of functions

.. math::

    d_n(x) = \sum_{k=1}^{n 2^n} \frac{k-1}{2^n} \mathbb{1} 
        \left\{ \frac{k-1}{2^n} \leq x < \frac{k}{2^n} \right\}
        + n \mathbb{1}\{ x \geq n \}


for :math:`n = 1, 2, \ldots`

Here :math:`\mathbb{1}\{ P \} = 1` if the statement :math:`P` is true and 0 otherwise

(This result is often used in measure theory)

Plot the functions :math:`d_n, \; n=1, \ldots, 6` on the interval :math:`[0, 10]` and compare them to the identity function

Do they get closer to the identity as :math:`n` gets larger?



Solutions
==========

Our aim is to plot the sequence of functions described in the exercise.
We will use the library ``Plots.jl``.

.. code-block:: julia

    Plots.pyplot()





Here's the function :math:`d_n` for any given :math:`n`:

.. code-block:: julia

    function d(x, n)
        current_val = 0
        for k in 1:(n * 2^n)
            if (k - 1) / 2^n <= x < k / 2^n
                current_val += (k - 1) / 2^n
            end
        end
        if x >= n
            current_val += n
        end
        return current_val
    end





.. code-block:: julia

    x_grid = linspace(0, 10, 100)
    n_vals = [1, 2, 3, 4, 5]
    
    function_vals = []
    labels = []
    
    for n in n_vals
        push!(function_vals, [d(x, n) for x in x_grid])
        push!(labels, "$n")
    end
    
    push!(function_vals, x_grid)
    push!(labels, "identity function")
    
    Plots.plot(x_grid, 
               function_vals, 
               label=reshape(labels, 1, length(n_vals) + 1),
               ylim=(0, 10))

