.. _covid_sciml:

.. include:: /_static/includes/header.raw

.. highlight:: julia

*******************************************************************
:index:`Parameter and Model Uncertainty in the COVID 19 Models`
*******************************************************************

.. contents:: :depth: 2

Overview
=============

Coauthored with Chris Rackauckas


Using the model developed in the :doc:`Modeling COVID 19 with (Stochastic) Differential Equations <../continuous_time/seir_model_sde>`, we explore further topics such as:

* Finding the derivatives of solutions to ODEs/SDEs
* Parameter uncertainty
* Bayesian estimation of the differential equations
* Minimizing loss functions with solutions to ODE/SDEs


Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics, Random, SparseArrays

.. code-block:: julia
    :class: Test

    using Test # Put this before any code in the lecture.

In addition, we will be exploring packages within the `SciML ecosystem <https://github.com/SciML/>`__ and
others covered in previous lectures 

.. code-block:: julia

    using OrdinaryDiffEq, StochasticDiffEq, DiffEqJump, DiffEqFlux, DiffEqBayes
    using Parameters, StaticArrays, Plots


ROUGH NOTES
==============

SOME OLDER NOTES:

The hope is that we have all or almost all of the model elements developed here, so that we can
focus on the machine learning/etc. in the following lecture and not implement any new model features (i.e. just shut things down as required).


Some thoughts on the SciML lecture which would be separate:

1.  Bayesian estimation and model uncertainty.  Have uncertainty on the parameter such as the :math:`m_0` parameter.  Show uncertainty that comes after the estimation of the parameter.
2.  Explain how Bayesian priors are a form of regularization.  With rare events like this, you will always have
more parameters than data.
3.  Solve the time-0 optimal policy problem using Flux of choosing the :math:`B` policy in some form.  For the objective, consider the uncertainty and the asymmetry of being wrong.
4. If we wanted to add in a model element, I think that putting in an asymptomatic state would be very helpful for explaining the role of uncertainty in evaluating the counter-factuals.
5. Solve for the optimal Markovian B(t) policy through simulation and using Flux. 