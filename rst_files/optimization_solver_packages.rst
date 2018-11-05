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
